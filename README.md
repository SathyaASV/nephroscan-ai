# NephroScan AI

Medical imaging AI platform for expo demonstration. Educational prototype only — not a medical device.

## Quick Start

### Local Development

```bash
pip install -r requirements.txt
python app.py
```

Server starts at `http://localhost:5000`. Open in browser.

### Docker

```bash
docker compose up --build
```

### Production (Render)

Push to GitHub. Render auto-deploys using `render.yaml`.

## Project Structure

```
NephroScan-AI/
├── app.py                 # Unified Flask server (API + frontend)
├── ai/gradcam.py          # Grad-CAM explainability
├── models/                # ResNet-18 .pth checkpoints
│   ├── kidney_stone_resnet18.pth
│   ├── chest_pneumonia_resnet18.pth
│   ├── brain_mri_resnet18.pth
│   └── heart_cardiomegaly_resnet18_improved.pth
├── frontend/index.html    # Single-page frontend
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── render.yaml
└── .env.example
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Server + model status |
| `/api/predict` | POST | Kidney stone classification |
| `/api/predict-chest` | POST | Chest pneumonia classification |
| `/api/predict-brain` | POST | Brain MRI classification |
| `/api/predict-heart` | POST | Heart cardiomegaly classification |
| `/api/explain` | POST | Grad-CAM attention heatmap |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | 5000 | Server port |
| `MODEL_DIR` | models | Model checkpoint directory |
| `CORS_ORIGINS` | * | Allowed CORS origins |
| `MAX_UPLOAD_BYTES` | 20971520 | Max upload size (20MB) |
| `INFERENCE_TIMEOUT` | 30 | Inference timeout (seconds) |

## Safety

- Every result includes provenance metadata (model, timestamp, inference type)
- Thermal proxy is labeled "VISUAL SIMULATION" — not an infrared measurement
- No clinical diagnosis — all results are AI-assisted screening prototypes
- See `DEMO_SCRIPT.md` for presentation guidelines
