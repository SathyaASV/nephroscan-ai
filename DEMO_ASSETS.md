# Demo Assets — Sources & Licenses

## Overview

NephroScan AI is a **research prototype and educational demo**. It does not store, process, or transmit real patient data. All demo images are either procedurally generated synthetic placeholders or sourced from public datasets under permissive licenses.

---

## Demo Image Inventory

| File | Scan Type | Source | License |
|------|-----------|--------|---------|
| `kidney_normal.png` | Kidney CT | Procedurally generated placeholder | Synthetic — no license required |
| `chest_normal.png` | Chest X-ray | Procedurally generated placeholder | Synthetic — no license required |
| `brain_tumor.png` | Brain MRI | Procedurally generated placeholder | Synthetic — no license required |
| `heart_cardio.png` | Heart X-ray | Procedurally generated placeholder | Synthetic — no license required |

---

## Real Dataset Samples (for backend inference testing only)

These images exist in the repository for testing the Python inference pipeline. They are **not** served to the frontend demo:

| Directory | Source | License |
|-----------|--------|---------|
| `chest_samples/` | [Chest X-ray dataset (Kermany et al., 2018)](https://data.mendeley.com/datasets/rscbj4/3) | CC BY 4.0 |
| `abdomen_samples_clear/` | [Abdomen CT samples](https://github.com/ Maheshkkorgi/CT-Scan-Images) | Public domain / CC0 |
| `my dataset final 512x512(implemented)/` | Custom synthetic dataset | Procedurally generated |

---

## Provenance Notes

- All demo images in `frontend/demo_assets/` are **synthetic placeholders** generated for layout and workflow demonstration. They do not represent real patient data.
- The backend inference pipeline (`ai/diagnose_gradcam.py`) is tested against real public dataset samples, which are excluded from deployment via `.gitignore`.
- No Protected Health Information (PHI) or Personally Identifiable Information (PII) is included in any asset.

---

## Regulatory Boundary

This application is a **research prototype only**. It is not:
- A medical device
- FDA/CE cleared
- Cleared for clinical use

All predictions are illustrative and must be reviewed by a qualified healthcare professional.
