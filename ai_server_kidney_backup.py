from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\models\kidney_stone_resnet18.pth"

device = torch.device("cpu")

# Load trained model
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

classes = checkpoint["classes"]
image_size = checkpoint.get("image_size", 128)

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()
model.to(device)

# Same preprocessing used during training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "NephroScan AI Model",
        "classes": classes
    })


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    try:

        image = Image.open(
            io.BytesIO(file.read())
        ).convert("RGB")

        image = transform(image)

        image = image.unsqueeze(0)

        with torch.no_grad():

            output = model(image)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        predicted_class = classes[
            prediction.item()
        ]

        confidence_percent = (
            confidence.item() * 100
        )

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(
                confidence_percent,
                2
            ),
            "classes": classes
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    print("----------------------------------------")
    print("       NephroScan AI Model Server")
    print("----------------------------------------")
    print("Model:", MODEL_PATH)
    print("Classes:", classes)
    print("Server: http://localhost:5000")
    print("Health: http://localhost:5000/health")
    print("Predict: http://localhost:5000/predict")
    print("----------------------------------------")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )