"""
NephroScan AI — Unified Production Server

Merges AI inference, frontend serving, and health endpoints into a single
Flask application. Designed for Gunicorn on Render or any PaaS.

    gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 180 app:app

Educational prototype only. Not a medical diagnostic device.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Optional OCR dependencies — gracefully degrade if absent
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    pytesseract = None
    OCR_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes as _pdf_to_images
    PDF_AVAILABLE = True
except ImportError:
    _pdf_to_images = None
    PDF_AVAILABLE = False

# Ensure ai/ subpackage is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "ai"))

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"
MODEL_DIR = ROOT / os.getenv("MODEL_DIR", "models")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "20971520"))
INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT", "30"))
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "1024"))

APP_VERSION = "2.1.0"

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_models(application: Flask) -> None:
    """Load all four ResNet-18 checkpoints at startup."""
    import logging
    log = logging.getLogger("nephroscan.models")

    device = torch.device("cpu")
    application.config["DEVICE"] = device
    application.config["STARTUP_TIME"] = datetime.now(timezone.utc).isoformat()

    log.info("=" * 60)
    log.info("NephroScan AI v%s — Loading models", APP_VERSION)
    log.info("Device: %s", device)
    log.info("MODEL_DIR resolved to: %s (exists=%s)", MODEL_DIR, MODEL_DIR.exists())
    log.info("=" * 60)

    model_specs = {
        "kidney": {
            "path": "kidney_stone_resnet18.pth",
            "calibrated": False,
            "threshold": None,
            "calibrated_label": None,
            "grayscale": True,
        },
        "chest": {
            "path": "chest_pneumonia_resnet18.pth",
            "calibrated": True,
            "threshold": 0.80,
            "calibrated_label": "pneumonia",
            "grayscale": False,
        },
        "brain": {
            "path": "brain_mri_resnet18.pth",
            "calibrated": False,
            "threshold": None,
            "calibrated_label": None,
            "grayscale": True,
        },
        "heart": {
            "path": "heart_cardiomegaly_resnet18_improved.pth",
            "calibrated": True,
            "threshold": 0.60,
            "calibrated_label": "cardiomegaly",
            "grayscale": False,
        },
    }

    models_loaded = {}
    NORMALIZE = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    for organ, spec in model_specs.items():
        model_path = MODEL_DIR / spec["path"]
        log.info("[%s] Loading %s …", organ, model_path)
        try:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {model_path} "
                    f"(dir contents: {[f.name for f in MODEL_DIR.iterdir()] if MODEL_DIR.exists() else 'DIR_MISSING'})"
                )
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            classes = checkpoint["classes"]
            image_size = checkpoint.get("image_size", 128)

            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(classes))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to("cpu")
            model.eval()

            transform_list = []
            if spec["grayscale"]:
                transform_list.append(transforms.Grayscale(num_output_channels=3))
            transform_list.extend([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                NORMALIZE,
            ])

            models_loaded[organ] = {
                "model": model,
                "classes": classes,
                "image_size": image_size,
                "transform": transforms.Compose(transform_list),
                "checkpoint_name": spec["path"],
                "calibrated": spec["calibrated"],
                "threshold": spec["threshold"],
                "calibrated_label": spec["calibrated_label"],
                "loaded": True,
            }
            log.info("[%s] OK — classes=%s, image_size=%d", organ, classes, image_size)
        except Exception as e:
            log.error("[%s] FAILED to load %s: %s", organ, spec["path"], e, exc_info=True)
            models_loaded[organ] = {
                "model": None,
                "classes": [],
                "image_size": 128,
                "transform": None,
                "checkpoint_name": spec["path"],
                "calibrated": spec["calibrated"],
                "threshold": spec["threshold"],
                "calibrated_label": spec["calibrated_label"],
                "loaded": False,
                "error": str(e),
            }

    loaded_count = sum(1 for m in models_loaded.values() if m["loaded"])
    log.info("=" * 60)
    log.info("Models loaded: %d / %d", loaded_count, len(models_loaded))
    log.info("=" * 60)

    application.config["MODELS"] = models_loaded
    application.config["DEVICE"] = device

    # Grad-CAM model map (only for loaded models)
    explain_map = {}
    for organ, data in models_loaded.items():
        if data["loaded"] and data["model"] is not None:
            explain_map[organ] = (
                data["model"],
                data["transform"],
                data["image_size"],
                data["checkpoint_name"].replace(".pth", ""),
            )
    application.config["EXPLAIN_MAP"] = explain_map


def _warmup_models(application: Flask) -> None:
    """Run a dummy inference on each loaded model to warm up CPU kernels."""
    import logging
    log = logging.getLogger("nephroscan.warmup")
    device = application.config["DEVICE"]
    models = application.config.get("MODELS", {})

    for organ, spec in models.items():
        if not spec.get("loaded") or spec["model"] is None:
            continue
        model = spec["model"]
        image_size = spec["image_size"]
        t0 = time.perf_counter()
        try:
            dummy = torch.randn(1, 3, image_size, image_size).to(device)
            with torch.inference_mode():
                _ = model(dummy)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            log.info("[%s] warm-up OK in %.1fms", organ, elapsed_ms)
        except Exception as e:
            log.warning("[%s] warm-up failed: %s", organ, e)


# ---------------------------------------------------------------------------
# Provenance helper
# ---------------------------------------------------------------------------

def _make_provenance(organ: str, application: Flask) -> dict:
    models = application.config.get("MODELS", {})
    m = models.get(organ, {})
    return {
        "model": m.get("checkpoint_name", "unknown"),
        "version": APP_VERSION,
        "inference_type": "REAL_MODEL_INFERENCE" if m.get("loaded") else "MODEL_UNAVAILABLE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(application.config.get("DEVICE", "unknown")),
        "preprocessing": f"resize-{m.get('image_size', 128)}-normalize",
    }


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def _predict_image(
    application: Flask,
    organ: str,
    file_storage,
) -> dict:
    import logging
    log = logging.getLogger("nephroscan.inference")
    timings = {}
    t_total_start = time.perf_counter()

    device = application.config["DEVICE"]
    models = application.config["MODELS"]
    spec = models[organ]

    if not spec["loaded"]:
        return {
            "error": f"Model for {organ} is not available",
            "provenance": _make_provenance(organ, application),
        }

    model = spec["model"]
    classes = spec["classes"]
    transform = spec["transform"]
    calibrated = spec["calibrated"]
    threshold = spec["threshold"]
    calibrated_label = spec["calibrated_label"]

    # Stage 1: Read & open image
    t0 = time.perf_counter()
    raw_bytes = file_storage.read()
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    timings["read_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Stage 2: Preprocess (resize, normalize, to tensor)
    t0 = time.perf_counter()
    image_size = spec["image_size"]
    if max(image.width, image.height) > MAX_IMAGE_DIM:
        image.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.LANCZOS)
    tensor = transform(image).unsqueeze(0).to(device)
    timings["preprocess_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Stage 3: Model inference
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    timings["inference_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # Stage 4: Postprocess
    t0 = time.perf_counter()
    original_index = int(torch.argmax(probabilities).item())
    original_prediction = classes[original_index]
    original_confidence = float(probabilities[original_index].item() * 100)

    if calibrated and threshold is not None:
        positive_prob = float(probabilities[1].item())
        prediction_index = 1 if positive_prob >= threshold else 0
    else:
        positive_prob = None
        prediction_index = original_index

    predicted_class = classes[prediction_index]
    confidence_percent = float(probabilities[prediction_index].item() * 100)
    timings["postprocess_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    result = {
        "prediction": predicted_class,
        "confidence": round(confidence_percent, 2),
        "classes": classes,
        "original_prediction": original_prediction,
        "original_confidence": round(original_confidence, 2),
        "threshold_calibrated": calibrated,
        "provenance": _make_provenance(organ, application),
    }

    if calibrated and positive_prob is not None:
        result["positive_probability"] = round(positive_prob * 100, 2)
        result["decision_threshold"] = threshold
        result["calibrated_label"] = calibrated_label

        if calibrated_label == "pneumonia":
            result["pneumonia_probability"] = round(positive_prob * 100, 2)
        if calibrated_label == "cardiomegaly":
            result["cardiomegaly_probability"] = round(positive_prob * 100, 2)

    # Stage 5: Response serialization (jsonify happens outside, just log total)
    timings["total_ms"] = round((time.perf_counter() - t_total_start) * 1000, 1)

    log.info(
        "[%s] predict timings: read=%.1fms preprocess=%.1fms inference=%.1fms postprocess=%.1fms total=%.1fms",
        organ,
        timings["read_ms"], timings["preprocess_ms"],
        timings["inference_ms"], timings["postprocess_ms"],
        timings["total_ms"],
    )

    result["timings"] = timings
    return result


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

ALLOWED_TYPES = {
    "image/jpeg", "image/png", "image/jpg",
    "application/pdf", "application/dicom", "image/dicom",
}


def _validate_upload(file_storage) -> str | None:
    """Return an error message if the upload is invalid, else None."""
    if file_storage is None:
        return "No image uploaded"

    content_type = file_storage.content_type or ""
    filename = (file_storage.filename or "").lower()
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""

    is_dicom = ext in ("dcm", "dicom")
    if content_type not in ALLOWED_TYPES and not is_dicom:
        return f"Unsupported file type: {content_type}. Accepted: JPG, PNG, PDF, DICOM"

    # Read and check size
    data = file_storage.read()
    if len(data) > MAX_UPLOAD_BYTES:
        return f"File too large: {len(data)} bytes (max {MAX_UPLOAD_BYTES})"
    if len(data) < 100:
        return "File appears empty or corrupted"

    # Reset stream so downstream can read it
    file_storage.seek(0)

    # Basic image integrity check
    if content_type.startswith("image/") and ext not in ("dcm", "dicom"):
        try:
            img = Image.open(io.BytesIO(data))
            img.verify()
        except Exception:
            return "Image file appears corrupted or unreadable"

    return None


# ---------------------------------------------------------------------------
# Lab Report Helpers
# ---------------------------------------------------------------------------

_LAB_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "application/pdf"}
_LAB_MAX_BYTES = 20 * 1024 * 1024  # 20 MB


def _lab_validate_upload(file_storage) -> str | None:
    """Return an error message if the lab upload is invalid, else None."""
    if file_storage is None:
        return "No file uploaded"
    content_type = file_storage.content_type or ""
    filename = (file_storage.filename or "").lower()
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
    if content_type not in _LAB_ALLOWED_TYPES:
        return f"Unsupported type: {content_type}. Accepted: JPG, PNG, PDF"
    data = file_storage.read()
    if len(data) > _LAB_MAX_BYTES:
        return f"File too large: {len(data)} bytes (max {_LAB_MAX_BYTES})"
    if len(data) < 100:
        return "File appears empty or corrupted"
    file_storage.seek(0)
    if content_type.startswith("image/"):
        try:
            img = Image.open(io.BytesIO(data))
            img.verify()
        except Exception:
            return "Image file appears corrupted or unreadable"
    return None


def _lab_image_from_upload(file_storage) -> Image.Image | None:
    """Convert an uploaded file (image or PDF first page) to a PIL Image."""
    content_type = file_storage.content_type or ""
    data = file_storage.read()
    file_storage.seek(0)
    if content_type == "application/pdf":
        if not PDF_AVAILABLE or _pdf_to_images is None:
            return None
        try:
            images = _pdf_to_images(data, first_page=1, last_page=1, dpi=300)
            return images[0].convert("RGB") if images else None
        except Exception:
            return None
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def _lab_ocr_image(img: Image.Image) -> str:
    """Run OCR on a PIL Image and return extracted text."""
    if not OCR_AVAILABLE or pytesseract is None:
        return ""
    try:
        text = pytesseract.image_to_string(img, lang="eng")
        return text or ""
    except Exception:
        return ""


# Common lab test patterns: "Test Name  12.3  g/dL  11.0-15.0"
# Matches: word chars, spaces, slashes, dots, parens for test names,
#          then numeric value, optional unit, optional range "low-high" or "<high" or ">low"
_LAB_TEST_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9 /(),.%-]{1,60})\s+"  # test name
    r"(\d+\.?\d*)\s+"                             # numeric value
    r"([A-Za-z/%μµ IU.-]{0,20})\s*"              # optional unit
    r"(?:(\d+\.?\d+)\s*[-–]\s*(\d+\.?\d+)|"      # range: low-high
    r"[<>≤≥]\s*(\d+\.?\d+))?",                     # or < / > threshold
    re.MULTILINE,
)

_LAB_DISCLAIMER = (
    "Educational decision support only. This is not a diagnosis "
    "and does not replace a qualified healthcare professional."
)


def _lab_parse_tests(text: str) -> list[dict]:
    """Extract lab test rows from OCR text. Never invent values."""
    tests = []
    for m in _LAB_TEST_RE.finditer(text):
        name = m.group(1).strip()
        value = m.group(2).strip()
        unit = (m.group(3) or "").strip()
        ref_low = (m.group(4) or "").strip()
        ref_high = (m.group(5) or m.group(6) or "").strip()

        # Skip obvious non-test lines
        lower_name = name.lower()
        if any(skip in lower_name for skip in (
            "patient", "name", "date", "time", "sample", "collected",
            "hospital", "lab ", "doctor", "physician", "report",
            "page", "total", "ref", "normal", "result", "status",
        )):
            continue

        status = "Needs review"
        if ref_low and ref_high:
            try:
                v, lo, hi = float(value), float(ref_low), float(ref_high)
                if v < lo:
                    status = "Below stated range"
                elif v > hi:
                    status = "Above stated range"
                else:
                    status = "Within stated range"
            except ValueError:
                status = "Needs review"

        tests.append({
            "name": name,
            "value": value,
            "unit": unit,
            "refLow": ref_low,
            "refHigh": ref_high,
            "status": status,
            "confidence": "OCR extraction",
        })
    return tests


def _lab_build_report(tests: list[dict], context: dict, filename: str) -> dict:
    """Build the full lab analysis report JSON from extracted tests."""
    problems = []
    details_for_summary = []
    for t in tests:
        if t["status"] == "Above stated range":
            problems.append(f"{t['name']} is above the stated reference range ({t['value']} {t['unit']}).")
            details_for_summary.append(f"{t['name']}: {t['value']} {t['unit']} (high)")
        elif t["status"] == "Below stated range":
            problems.append(f"{t['name']} is below the stated reference range ({t['value']} {t['unit']}).")
            details_for_summary.append(f"{t['name']}: {t['value']} {t['unit']} (low)")

    if problems:
        summary = (
            f"Analysis of {filename} identified {len(problems)} value(s) outside "
            f"the laboratory's printed reference range: "
            + "; ".join(details_for_summary[:5])
            + ". A qualified clinician should review these results in the context of your clinical history."
        )
    elif tests:
        summary = (
            f"Analysis of {filename} found {len(tests)} test value(s), "
            "all within the laboratory's printed reference ranges. "
            "This does not rule out all conditions — share the full report with your clinician."
        )
    else:
        summary = (
            f"Analysis of {filename} could not extract structured test values. "
            "The report may be handwritten, low-resolution, or in an unsupported format. "
            "Please upload a clearer image or PDF, or bring the original to your clinician."
        )

    what_can_be_done = [
        "Share this report with a qualified clinician for interpretation.",
        "Bring a copy of the original lab report to your next appointment.",
        "If values are flagged, ask your clinician whether repeat testing is advised.",
        "Note any symptoms, medications, or recent changes in health to discuss.",
    ]

    diet_guidance = [
        "General balanced nutrition supports overall health — no specific diet changes are recommended based on lab values alone.",
        "Stay well-hydrated unless otherwise advised by your clinician.",
        "Discuss any dietary supplements or changes with your healthcare provider before making them.",
    ]

    lifestyle_guidance = [
        "Maintain regular sleep patterns (7-9 hours for most adults).",
        "Stay physically active as tolerated and as advised by your clinician.",
        "Avoid smoking and limit alcohol consumption.",
        "Keep a record of symptoms, medications, and lifestyle changes for your clinician review.",
    ]

    urgency = "Routine follow-up with your clinician is recommended to discuss these results."
    if problems:
        urgency = (
            "Some values are outside the stated reference range. "
            "Prompt review by a clinician is recommended, especially if you have symptoms. "
            "Seek urgent care if you experience severe symptoms."
        )

    discussion = [
        "What do these results mean in the context of my symptoms and medical history?",
        "Are any follow-up tests needed to confirm or investigate these findings?",
        "Should any medications or supplements be adjusted based on these results?",
        "When should I schedule a follow-up appointment?",
    ]

    uncertainty = []
    if not tests:
        uncertainty.append("No structured test values could be extracted from the uploaded report.")
    fasting = context.get("fasting", "")
    if not fasting:
        uncertainty.append("Fasting status is unknown — some tests may require fasting for accurate interpretation.")
    symptoms = context.get("symptoms", "")
    if not symptoms:
        uncertainty.append("No symptom information provided — clinical correlation is essential.")
    medicines = context.get("medicines", "")
    if not medicines:
        uncertainty.append("Medication history is not available — some drugs can affect lab values.")
    pregnancy = context.get("pregnancyStatus", "")
    if not pregnancy:
        uncertainty.append("Pregnancy status is unknown — reference ranges may differ during pregnancy.")

    return {
        "status": "ok",
        "tests": tests,
        "overallSummary": summary,
        "possibleProblems": problems,
        "whatCanBeDone": what_can_be_done,
        "dietGuidance": diet_guidance,
        "lifestyleGuidance": lifestyle_guidance,
        "urgencyGuidance": urgency,
        "doctorDiscussionPoints": discussion,
        "uncertainty": uncertainty,
        "disclaimer": _LAB_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

GRADCAM_DISCLAIMER = (
    "Attention visualization, not a lesion segmentation or diagnosis. "
    "Highlighted regions show where the model weighted its decision — "
    "they do not confirm the presence or location of disease."
)


def _register_routes(application: Flask) -> None:

    # ---- Health ----

    @application.route("/api/health", methods=["GET"])
    def api_health():
        models = application.config.get("MODELS", {})
        model_status = {}
        all_loaded = True
        for organ, data in models.items():
            model_status[organ] = {
                "loaded": data.get("loaded", False),
                "checkpoint": data.get("checkpoint_name", "unknown"),
                "classes": data.get("classes", []),
            }
            if not data.get("loaded"):
                all_loaded = False

        return jsonify({
            "status": "online",
            "service": "NephroScan AI",
            "version": APP_VERSION,
            "device": str(application.config.get("DEVICE", "unknown")),
            "models": model_status,
            "all_models_loaded": all_loaded,
            "startup_time": application.config.get("STARTUP_TIME"),
            "endpoints": [
                "/api/health",
                "/api/predict",
                "/api/predict-chest",
                "/api/predict-brain",
                "/api/predict-heart",
                "/api/explain",
                "/api/lab/health",
                "/api/lab/analyze",
            ],
        })

    # ---- Predict (kidney) ----

    @application.route("/api/predict", methods=["POST"])
    def api_predict_kidney():
        error = _validate_upload(request.files.get("image"))
        if error:
            return jsonify({"error": error}), 400
        try:
            result = _predict_image(application, "kidney", request.files["image"])
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e), "provenance": _make_provenance("kidney", application)}), 500

    # ---- Predict chest ----

    @application.route("/api/predict-chest", methods=["POST"])
    def api_predict_chest():
        error = _validate_upload(request.files.get("image"))
        if error:
            return jsonify({"error": error}), 400
        try:
            result = _predict_image(application, "chest", request.files["image"])
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e), "provenance": _make_provenance("chest", application)}), 500

    # ---- Predict brain ----

    @application.route("/api/predict-brain", methods=["POST"])
    def api_predict_brain():
        error = _validate_upload(request.files.get("image"))
        if error:
            return jsonify({"error": error}), 400
        try:
            result = _predict_image(application, "brain", request.files["image"])
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e), "provenance": _make_provenance("brain", application)}), 500

    # ---- Predict heart ----

    @application.route("/api/predict-heart", methods=["POST"])
    def api_predict_heart():
        error = _validate_upload(request.files.get("image"))
        if error:
            return jsonify({"error": error}), 400
        try:
            result = _predict_image(application, "heart", request.files["image"])
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e), "provenance": _make_provenance("heart", application)}), 500

    # ---- Explain (Grad-CAM) ----

    @application.route("/api/explain", methods=["POST"])
    def api_explain():
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        scan_type = request.form.get("scan_type", "").strip().lower()
        explain_map = application.config.get("EXPLAIN_MAP", {})
        entry = explain_map.get(scan_type)

        if entry is None:
            return jsonify({
                "status": "unavailable",
                "error": f"Unsupported scan_type '{scan_type}'. Expected: kidney, chest, brain, heart.",
                "disclaimer": GRADCAM_DISCLAIMER,
            }), 400

        from gradcam import generate_gradcam, overlay_to_base64_png

        model, transform, image_size, model_name = entry

        try:
            pil_image = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Could not read image: {e}"}), 400

        try:
            result = generate_gradcam(model, transform, image_size, pil_image)
            classes = application.config["MODELS"][scan_type]["classes"]
            predicted_class = classes[result["predicted_index"]]
            heatmap_b64 = overlay_to_base64_png(result["overlay_image"])

            return jsonify({
                "status": "ok",
                "scan_type": scan_type,
                "model": model_name,
                "prediction": predicted_class,
                "heatmap_image": heatmap_b64,
                "disclaimer": GRADCAM_DISCLAIMER,
            })
        except Exception:
            return jsonify({
                "status": "unavailable",
                "model": model_name,
                "message": "Attention visualization could not be generated for this image.",
                "disclaimer": GRADCAM_DISCLAIMER,
            }), 200

    # ---- Lab Report Analysis ----

    @application.route("/api/lab/health", methods=["GET"])
    def api_lab_health():
        return jsonify({
            "status": "ok",
            "lab_endpoint": True,
            "version": "lab-route-1",
            "ocr_available": OCR_AVAILABLE,
            "pdf_support": PDF_AVAILABLE,
            "endpoint": "/api/lab/analyze",
        })

    @application.route("/api/lab/analyze", methods=["POST"])
    def api_lab_analyze():
        error = _lab_validate_upload(request.files.get("lab_report"))
        if error:
            return jsonify({"status": "error", "message": error}), 400

        file_storage = request.files["lab_report"]
        filename = file_storage.filename or "lab_report"
        content_type = file_storage.content_type or ""

        # Parse optional context JSON
        context = {}
        ctx_raw = request.form.get("context", "")
        if ctx_raw:
            try:
                import json as _json
                context = _json.loads(ctx_raw)
            except Exception:
                context = {}

        # Convert to image (handles both images and PDFs)
        img = _lab_image_from_upload(file_storage)
        if img is None:
            return jsonify({
                "status": "error",
                "message": "Could not process the uploaded file. Ensure it is a valid image or PDF.",
            }), 400

        # OCR
        raw_text = _lab_ocr_image(img)
        if not raw_text or len(raw_text.strip()) < 10:
            return jsonify({
                "status": "ok",
                "tests": [],
                "overallSummary": (
                    f"OCR could not extract readable text from {filename}. "
                    "The report may be handwritten, low-resolution, or in an unsupported format. "
                    "Please upload a clearer image or PDF."
                ),
                "possibleProblems": [],
                "whatCanBeDone": [
                    "Upload a clearer, high-resolution image or PDF of the lab report.",
                    "Ensure the image is well-lit and in focus.",
                    "Bring the original paper report to your clinician.",
                ],
                "dietGuidance": [],
                "lifestyleGuidance": [],
                "urgencyGuidance": "Unable to assess urgency from the uploaded file. Consult your clinician.",
                "doctorDiscussionPoints": [],
                "uncertainty": ["OCR failed or produced insufficient readable text."],
                "disclaimer": _LAB_DISCLAIMER,
            })

        # Parse test rows
        tests = _lab_parse_tests(raw_text)

        # Build full report
        report = _lab_build_report(tests, context, filename)
        return jsonify(report)

    # ---- Frontend serving ----

    @application.route("/")
    def serve_index():
        index_path = FRONTEND_DIR / "index.html"
        if not index_path.exists():
            return jsonify({"error": "Frontend not found"}), 500
        return send_from_directory(str(FRONTEND_DIR), "index.html", mimetype="text/html")

    @application.route("/<path:filename>")
    def serve_static(filename: str):
        if filename.startswith("api/"):
            return jsonify({"error": "Not found"}), 404
        return send_from_directory(str(FRONTEND_DIR), filename)

    # ---- Catch-all for SPA routing ----

    @application.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404


# ---------------------------------------------------------------------------
# Application factory (Gunicorn-compatible)
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    # Prevent Gunicorn fork-deadlock: single-threaded CPU inference
    torch.set_num_threads(1)

    application = Flask(__name__, static_folder=None)

    # Narrow CORS from environment in production
    CORS(application, origins=CORS_ORIGINS.split(","), supports_credentials=True)

    _load_models(application)
    _warmup_models(application)

    _register_routes(application)

    return application


app = create_app()


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"NephroScan AI v{APP_VERSION}")
    print(f"Server: http://0.0.0.0:{port}")
    print(f"Health: http://0.0.0.0:{port}/api/health")
    app.run(host="0.0.0.0", port=port, debug=False)
