# Demo Assets — Sources & Licenses

## Overview

NephroScan AI is a **research prototype and educational demo**. It does not store, process, or transmit real patient data. Demo images are sourced from public datasets under permissive licenses or are procedurally generated synthetic placeholders.

---

## Demo Case Images (real model inference)

These images are loaded by the frontend "Load Demo Case" button and sent to the backend for real inference.

| File | Scan Type | Source | License |
|------|-----------|--------|---------|
| `chest_normal_real.png` | Chest X-ray | [Chest X-ray dataset (Kermany et al., 2018)](https://data.mendeley.com/datasets/rscbj4/3) | CC BY 4.0 |
| `brain_glioma_real.jpg` | Brain MRI | [Brain MRI dataset (Kermany et al., 2018)](https://data.mendeley.com/datasets/rscbj4/3) | CC BY 4.0 |
| `heart_cardio_real.png` | Chest X-ray (cardiomegaly) | [Chest X-ray dataset (Kermany et al., 2018)](https://data.mendeley.com/datasets/rscbj4/3) | CC BY 4.0 |

No kidney demo case is provided — no verified real kidney image was available.

---

## Synthetic Demo Images (precomputed results, no model call)

These images are used by the "Synthetic Demo" flow, which displays precomputed results without calling the model.

| File | Scan Type | Source | License |
|------|-----------|--------|---------|
| `kidney_stone.png` | Kidney CT | Procedurally generated placeholder | Synthetic — no license required |
| `chest_pneumonia.png` | Chest X-ray | Procedurally generated placeholder | Synthetic — no license required |
| `brain_tumor.png` | Brain MRI | Procedurally generated placeholder | Synthetic — no license required |
| `heart_cardio.png` | Heart X-ray | Procedurally generated placeholder | Synthetic — no license required |

---

## Provenance Notes

- Real demo images in `frontend/demo_assets/` are sourced from publicly available medical imaging datasets under CC BY 4.0 licenses.
- Synthetic placeholder images are procedurally generated for layout demonstration only.
- No Protected Health Information (PHI) or Personally Identifiable Information (PII) is included in any asset.

---

## Regulatory Boundary

This application is a **research prototype only**. It is not:
- A medical device
- FDA/CE cleared
- Cleared for clinical use

All predictions are illustrative and must be reviewed by a qualified healthcare professional.
