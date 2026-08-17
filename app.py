"""
NephroScan AI — Unified Production Server

Merges AI inference, frontend serving, and health endpoints into a single
Flask application. Designed for Gunicorn on Render or any PaaS.

    gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app

Educational prototype only. Not a medical diagnostic device.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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

APP_VERSION = "2.0.0"

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

    image = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

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

    application = Flask(__name__, static_folder=None)

    # Narrow CORS from environment in production
    CORS(application, origins=CORS_ORIGINS.split(","), supports_credentials=True)

    _load_models(application)

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
