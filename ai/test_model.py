import torch
from torchvision import transforms, models
from PIL import Image
import os

# =========================
# SETTINGS
# =========================

MODEL_PATH = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\models\kidney_stone_resnet18.pth"

IMAGE_PATH = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\TEST_IMAGE.jpg"

IMAGE_SIZE = 128

# =========================
# DEVICE
# =========================

device = torch.device("cpu")

# =========================
# LOAD CHECKPOINT
# =========================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

classes = checkpoint["classes"]

print("Classes:", classes)

# =========================
# CREATE MODEL
# =========================

model = models.resnet18(weights=None)

model.fc = torch.nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# CHECK IMAGE
# =========================

if not os.path.exists(IMAGE_PATH):

    print()
    print("ERROR: Test image not found.")
    print()
    print("Put an image in the NephroScan-AI folder")
    print("and name it:")
    print("TEST_IMAGE.jpg")
    print()

    raise SystemExit

# =========================
# PREDICTION
# =========================

image = Image.open(IMAGE_PATH)

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

# =========================
# RESULT
# =========================

print()
print("=" * 45)
print("NEPHROSCAN AI TEST RESULT")
print("=" * 45)

print(
    f"Prediction : {predicted_class}"
)

print(
    f"Confidence : {confidence_percent:.2f}%"
)

print("=" * 45)