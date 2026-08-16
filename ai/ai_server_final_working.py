import io
import os

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from gradcam import generate_gradcam, overlay_to_base64_png


# ----------------------------------------
# NephroScan AI model server
# Educational prototype only. Not a medical diagnosis.
# ----------------------------------------

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

KIDNEY_MODEL_PATH = os.path.join(MODELS_DIR, "kidney_stone_resnet18.pth")
CHEST_MODEL_PATH = os.path.join(MODELS_DIR, "chest_pneumonia_resnet18.pth")
BRAIN_MODEL_PATH = os.path.join(MODELS_DIR, "brain_mri_resnet18.pth")
HEART_MODEL_PATH = os.path.join(
    MODELS_DIR, "heart_cardiomegaly_resnet18_improved.pth"
)


def load_model(model_path):
    """Load a checkpoint created by the NephroScan training scripts."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 128)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes, image_size


# Load the four model checkpoints once when the server starts.
kidney_model, kidney_classes, kidney_image_size = load_model(
    KIDNEY_MODEL_PATH
)
chest_model, chest_classes, chest_image_size = load_model(
    CHEST_MODEL_PATH
)
brain_model, brain_classes, brain_image_size = load_model(
    BRAIN_MODEL_PATH
)
heart_model, heart_classes, heart_image_size = load_model(
    HEART_MODEL_PATH
)


NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

kidney_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((kidney_image_size, kidney_image_size)),
    transforms.ToTensor(),
    NORMALIZE,
])

chest_transform = transforms.Compose([
    transforms.Resize((chest_image_size, chest_image_size)),
    transforms.ToTensor(),
    NORMALIZE,
])

brain_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((brain_image_size, brain_image_size)),
    transforms.ToTensor(),
    NORMALIZE,
])

heart_transform = transforms.Compose([
    transforms.Resize((heart_image_size, heart_image_size)),
    transforms.ToTensor(),
    NORMALIZE,
])


# Explainability is limited to the four approved modalities only.
# Abdomen intentionally excluded — do not add it here.
EXPLAIN_MODEL_MAP = {
    "kidney": (kidney_model, kidney_transform, kidney_image_size, "kidney_stone_resnet18"),
    "chest":  (chest_model,  chest_transform,  chest_image_size,  "chest_pneumonia_resnet18"),
    "brain":  (brain_model,  brain_transform,  brain_image_size,  "brain_mri_resnet18"),
    "heart":  (heart_model,  heart_transform,  heart_image_size,  "heart_cardiomegaly_resnet18_improved"),
}

GRADCAM_DISCLAIMER = (
    "Attention visualization, not a lesion segmentation or diagnosis. "
    "Highlighted regions show where the model weighted its decision — "
    "they do not confirm the presence or location of disease."
)


def predict_image(
    file,
    model,
    classes,
    transform,
    calibrated=False,
    decision_threshold=0.80,
    calibrated_label="pneumonia"
):
    """Predict one image, optionally using class-1 threshold calibration."""
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    original_prediction_index = int(torch.argmax(probabilities).item())
    original_prediction = classes[original_prediction_index]
    original_confidence = float(
        probabilities[original_prediction_index].item() * 100
    )

    if calibrated:
        positive_probability = float(probabilities[1].item())
        prediction_index = (
            1 if positive_probability >= decision_threshold else 0
        )
    else:
        positive_probability = None
        prediction_index = original_prediction_index

    predicted_class = classes[prediction_index]
    confidence_percent = float(
        probabilities[prediction_index].item() * 100
    )

    result = {
        "prediction": predicted_class,
        "confidence": round(confidence_percent, 2),
        "classes": classes,
        "original_prediction": original_prediction,
        "original_confidence": round(original_confidence, 2),
        "threshold_calibrated": calibrated,
    }

    if calibrated:
        result["positive_probability"] = round(
            positive_probability * 100, 2
        )
        result["decision_threshold"] = decision_threshold
        result["calibrated_label"] = calibrated_label

        # Preserve the existing Chest frontend/backend field name.
        if calibrated_label == "pneumonia":
            result["pneumonia_probability"] = round(
                positive_probability * 100, 2
            )

        # Heart-specific readable field for future frontend use.
        if calibrated_label == "cardiomegaly":
            result["cardiomegaly_probability"] = round(
                positive_probability * 100, 2
            )

    return result


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "NephroScan AI Model",
        "device": str(device),
        "kidney_classes": kidney_classes,
        "chest_classes": chest_classes,
        "brain_classes": brain_classes,
        "heart_classes": heart_classes,
        "endpoints": [
            "/health",
            "/predict",
            "/predict-chest",
            "/predict-brain",
            "/predict-heart",
            "/explain",
        ],
    })


@app.route("/predict", methods=["POST"])
def predict_kidney():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        result = predict_image(
            request.files["image"],
            kidney_model,
            kidney_classes,
            kidney_transform,
        )
        result["model"] = "kidney_stone_resnet18"
        return jsonify(result)
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/predict-chest", methods=["POST"])
def predict_chest():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        result = predict_image(
            request.files["image"],
            chest_model,
            chest_classes,
            chest_transform,
            calibrated=True,
            decision_threshold=0.80,
            calibrated_label="pneumonia",
        )
        result["model"] = "chest_pneumonia_resnet18"
        return jsonify(result)
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/predict-brain", methods=["POST"])
def predict_brain():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        result = predict_image(
            request.files["image"],
            brain_model,
            brain_classes,
            brain_transform,
        )
        result["model"] = "brain_mri_resnet18"
        return jsonify(result)
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/predict-heart", methods=["POST"])
def predict_heart():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        result = predict_image(
            request.files["image"],
            heart_model,
            heart_classes,
            heart_transform,
            calibrated=True,
            decision_threshold=0.60,
            calibrated_label="cardiomegaly",
        )
        result["model"] = "heart_cardiomegaly_resnet18_improved"
        return jsonify(result)
    except Exception as error:
        return jsonify({"error": str(error)}), 500


@app.route("/explain", methods=["POST"])
def explain():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    scan_type = request.form.get("scan_type", "").strip().lower()
    entry = EXPLAIN_MODEL_MAP.get(scan_type)
    if entry is None:
        return jsonify({
            "status": "unavailable",
            "error": (
                f"Unsupported scan_type '{scan_type}'. "
                "Expected one of: kidney, chest, brain, heart."
            ),
            "disclaimer": GRADCAM_DISCLAIMER,
        }), 400

    model, transform, image_size, model_name = entry

    try:
        pil_image = Image.open(
            io.BytesIO(request.files["image"].read())
        ).convert("RGB")
    except Exception as error:
        return jsonify({"error": f"Could not read image: {error}"}), 400

    try:
        result = generate_gradcam(model, transform, image_size, pil_image)
        classes = {
            "kidney": kidney_classes,
            "chest": chest_classes,
            "brain": brain_classes,
            "heart": heart_classes,
        }[scan_type]
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
    except Exception as error:
        # Graceful fallback — never crash the server, never touch
        # the original /predict* results.
        return jsonify({
            "status": "unavailable",
            "model": model_name,
            "message": "Attention visualization could not be generated for this image.",
            "disclaimer": GRADCAM_DISCLAIMER,
        }), 200


if __name__ == "__main__":
    print("----------------------------------------")
    print("       NephroScan AI Four-Model Server")
    print("----------------------------------------")
    print("Device:", device)
    print("Kidney model:", KIDNEY_MODEL_PATH)
    print("Kidney classes:", kidney_classes)
    print("Chest model:", CHEST_MODEL_PATH)
    print("Chest classes:", chest_classes)
    print("Brain model:", BRAIN_MODEL_PATH)
    print("Brain classes:", brain_classes)
    print("Heart model:", HEART_MODEL_PATH)
    print("Heart classes:", heart_classes)
    print("Heart threshold: 0.60")
    print("Server: http://localhost:5000")
    print("Health: http://localhost:5000/health")
    print("Kidney predict: http://localhost:5000/predict")
    print("Chest predict: http://localhost:5000/predict-chest")
    print("Brain predict: http://localhost:5000/predict-brain")
    print("Heart predict: http://localhost:5000/predict-heart")
    print("Explain: http://localhost:5000/explain")
    print("----------------------------------------")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
