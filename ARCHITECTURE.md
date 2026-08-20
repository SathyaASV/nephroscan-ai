# NephroScan AI — Architecture

## Overview

Single-server Flask application serving both API and frontend. Designed for Render deployment with Gunicorn.

```
Browser (index.html)
    │
    ├── GET  /                    → serves index.html
    ├── GET  /api/health          → model status JSON
    ├── POST /api/predict         → kidney classification
    ├── POST /api/predict-chest   → chest classification
    ├── POST /api/predict-brain   → brain classification
    ├── POST /api/predict-heart   → heart classification
    ├── POST /api/explain         → Grad-CAM heatmap
    ├── POST /api/lab/analyze     → lab report OCR + guidance
    ├── GET  /api/ai/health       → optional AI layer status
    ├── POST /api/ai/analyze-image→ optional AI image description
    └── POST /api/ai/chat         → optional AI report Q&A
```

## Backend (`app.py`)

- **Factory pattern:** `create_app()` for Gunicorn compatibility
- **Model loading:** All 4 ResNet-18 checkpoints loaded at startup into `app.config["MODELS"]`
- **Inference pipeline:** Image → PIL → transform → tensor → model → softmax → result
- **Calibrated thresholds:** Chest (0.80) and heart (0.60) use post-hoc threshold calibration
- **Provenance:** Every response includes model name, version, timestamp, device, and inference type
- **Upload validation:** File type, size (`MAX_UPLOAD_BYTES`, 8 MB default), image integrity checks

## Optional Gemini Intelligence Layer (`/api/ai/*`)

The local classifiers only cover kidney, chest, brain, and heart scans. The
Gemini layer covers the rest — describing an out-of-scope image and answering
questions about a report — and is inert unless `AI_API_KEY` is set.

```
Browser ──same-origin POST──► Flask /api/ai/*
                                 │  validate + bound + re-encode
                                 │  (key never leaves the server)
                                 ▼
             Gemini OpenAI-compatible /chat/completions
```

- **Disabled by default:** without a key both POST routes return
  `503 {"status":"disabled","code":"ai_disabled"}` and the frontend keeps
  using local model output and offline guidance. `GET /api/ai/health` exposes the flag.
- **Transport:** one bounded outbound HTTPS call per request via `requests`
  (imported lazily), using Gemini's OpenAI-compatible chat-completions endpoint.
  `AI_BASE_URL` keeps the provider swappable without browser changes. No extra
  process, thread, or resident model — one worker on a 1 GB container is enough.
- **Image path:** JPG/PNG/WEBP only, ≤ `AI_MAX_IMAGE_BYTES`, verified with PIL,
  then re-encoded to a ≤ `AI_MAX_IMAGE_DIM` JPEG (this also strips EXIF) and
  sent as a base64 data URL. The encoded copy is dropped and `gc.collect()` runs
  as soon as the call returns.
- **Chat path:** JSON body ≤ `AI_MAX_JSON_BYTES`; roles restricted to
  `user`/`assistant`; the newest `AI_MAX_MESSAGES` turns are forwarded under a
  `AI_MAX_TOTAL_CHARS` budget; control characters are stripped. Caller context is
  serialised, truncated to `AI_MAX_CONTEXT_CHARS`, and injected as untrusted data.
- **Prompt safety:** shared rules forbid diagnosis, certainty, and prescribing;
  they require explicit uncertainty, clinician review, and immediate emergency
  care for red-flag symptoms. Every response carries the educational disclaimer.
- **No fabrication:** `analyze-image` requires a parseable JSON object with a
  non-empty `summary`; malformed provider output becomes `502 upstream_malformed`
  with no clinical fields. Provider faults map to explicit error codes.
- **Observability:** each request gets an `ai_<hex>` request id returned to the
  client and logged with route, byte count, model, and latency. Image bytes,
  message text, and credentials are never logged.

## Frontend (`frontend/index.html` + `frontend/src`)

The existing HTML shell keeps the proven imaging, lab, history, and demo flows. A strict TypeScript module under `frontend/src` is compiled to `frontend/dist/main.js` and loaded after the legacy handlers. Railway builds it in the Docker multi-stage build; local development runs `cd frontend && npm ci && npm run build`.

### Views
- **Dashboard** — Model status, session stats, quick actions
- **New Analysis** — Upload images, select scan type, run inference
- **History** — Session reports with filtering
- **Patients** — Patient management (demo)
- **Compare** — Side-by-side result comparison
- **Performance** — Model accuracy metrics
- **Assistant** — NephroBot clinical Q&A
- **Settings** — Theme, thresholds, export
- **Expo Presence** — Live camera with thermal proxy

### Expo Presence Pipeline
1. Browser `getUserMedia()` captures webcam feed
2. RGB canvas captures frames
3. Thermal proxy canvas applies colormap (luminance → heat gradient)
4. Simple motion heuristic estimates presence confidence
5. Results logged to session table

## Models

| Model | File | Classes | Calibrated |
|---|---|---|---|
| Kidney Stone | `kidney_stone_resnet18.pth` | stone/no_stone | No |
| Chest Pneumonia | `chest_pneumonia_resnet18.pth` | normal/pneumonia | Yes (0.80) |
| Brain MRI | `brain_mri_resnet18.pth` | tumor/no_tumor | No |
| Heart Cardiomegaly | `heart_cardiomegaly_resnet18_improved.pth` | normal/cardiomegaly | Yes (0.60) |

## Deployment

- **Platform:** Railway Trial or Render
- **Railway config:** `railway.json` selects the Dockerfile, `/api/health` healthcheck, one-worker start command, and bounded restart policy
- **Runtime:** Gunicorn with 1 worker, 1 thread, 180s timeout; Railway injects `$PORT`
- **Docker:** Node build stage compiles `frontend/src` to `frontend/dist/main.js`; Python 3.11-slim runtime serves the Flask app
- **Memory:** four local ResNet checkpoints are loaded once per worker; local smoke measurement was ~495 MiB after warm-up and ~499 MiB after one prediction. Trial's 1 GB service limit still requires one replica/worker and headroom for concurrent requests.
- **Auto-deploy:** Railway can deploy from the linked project with `railway up`; Render can rebuild from GitHub using `render.yaml`
