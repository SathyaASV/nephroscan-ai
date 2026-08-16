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
    └── POST /api/explain         → Grad-CAM heatmap
```

## Backend (`app.py`)

- **Factory pattern:** `create_app()` for Gunicorn compatibility
- **Model loading:** All 4 ResNet-18 checkpoints loaded at startup into `app.config["MODELS"]`
- **Inference pipeline:** Image → PIL → transform → tensor → model → softmax → result
- **Calibrated thresholds:** Chest (0.80) and heart (0.60) use post-hoc threshold calibration
- **Provenance:** Every response includes model name, version, timestamp, device, and inference type
- **Upload validation:** File type, size (20MB max), image integrity checks

## Frontend (`frontend/index.html`)

Single HTML file with embedded CSS and JavaScript. No build step.

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

- **Platform:** Render (free tier)
- **Runtime:** Gunicorn with 1 worker, 120s timeout
- **Docker:** Python 3.11-slim base image
- **Auto-deploy:** Push to GitHub → Render rebuilds
