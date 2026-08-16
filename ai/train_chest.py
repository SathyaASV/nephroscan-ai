import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist import PneumoniaMNIST

# =========================
# PATHS
# =========================

DATA_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\chest_data"
MODEL_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# SETTINGS
# =========================

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5

# CPU is safe for beginners. We can optimize later if needed.
device = torch.device("cpu")
print("Using device:", device)

# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD PNEUMONIA DATASET
# =========================

print("Loading PneumoniaMNIST...")

train_dataset = PneumoniaMNIST(
    split="train",
    root=DATA_DIR,
    transform=transform,
    as_rgb=True,
    download=False
)

val_dataset = PneumoniaMNIST(
    split="val",
    root=DATA_DIR,
    transform=transform,
    as_rgb=True,
    download=False
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Classes: normal and pneumonia")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =========================
# RESNET-18 MODEL
# =========================

print("Loading ResNet-18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze the pretrained feature extractor.
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer for two classes.
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

best_accuracy = 0.0
best_weights = copy.deepcopy(model.state_dict())

# =========================
# TRAINING
# =========================

for epoch in range(EPOCHS):
    print()
    print("=" * 40)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 40)

    model.train()
    train_total = 0
    train_correct = 0
    train_loss_total = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_total += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        train_total += labels.size(0)
        train_correct += (predictions == labels).sum().item()

    train_loss = train_loss_total / train_total
    train_accuracy = train_correct / train_total

    model.eval()
    val_total = 0
    val_correct = 0
    val_loss_total = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            val_total += labels.size(0)
            val_correct += (predictions == labels).sum().item()

    val_loss = val_loss_total / val_total
    val_accuracy = val_correct / val_total

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_weights = copy.deepcopy(model.state_dict())

# =========================
# SAVE MODEL
# =========================

model.load_state_dict(best_weights)

model_path = os.path.join(
    MODEL_DIR,
    "chest_pneumonia_resnet18.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": ["normal", "pneumonia"],
        "image_size": IMAGE_SIZE
    },
    model_path
)

print()
print("=" * 40)
print("CHEST MODEL TRAINING COMPLETE")
print("=" * 40)
print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")
print("Model saved to:")
print(model_path)
