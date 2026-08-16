import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

# =========================
# PATHS
# =========================

DATASET_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\my dataset final 512x512(implemented)"
MODEL_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# FAST SETTINGS
# =========================

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 4

MAX_IMAGES = 2500

device = torch.device("cpu")

print("Using device:", device)

# =========================
# TRANSFORMS
# =========================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD DATASET
# =========================

dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=transform
)

print("Classes:", dataset.classes)
print("Full dataset:", len(dataset))

# =========================
# USE SUBSET FOR SPEED
# =========================

if len(dataset) > MAX_IMAGES:

    indices = torch.randperm(
        len(dataset),
        generator=torch.Generator().manual_seed(42)
    )[:MAX_IMAGES].tolist()

    dataset = Subset(dataset, indices)

print("Images used:", len(dataset))

# =========================
# TRAIN / VALIDATION SPLIT
# =========================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

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

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

# =========================
# RESNET-18
# =========================

print("Loading ResNet-18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze pretrained layers

for param in model.parameters():
    param.requires_grad = False

# Replace classifier

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(dataset.dataset.classes)
)

model = model.to(device)

# =========================
# TRAINING
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

best_accuracy = 0

best_weights = copy.deepcopy(
    model.state_dict()
)

for epoch in range(EPOCHS):

    print()
    print("=" * 40)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 40)

    # TRAIN

    model.train()

    total = 0
    correct = 0
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_loss = running_loss / total

    train_accuracy = correct / total

    # VALIDATION

    model.eval()

    total = 0
    correct = 0
    val_loss_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item() *
                images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    val_loss = val_loss_total / total

    val_accuracy = correct / total

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_weights = copy.deepcopy(
            model.state_dict()
        )

# =========================
# SAVE MODEL
# =========================

model.load_state_dict(
    best_weights
)

model_path = os.path.join(
    MODEL_DIR,
    "kidney_stone_resnet18.pth"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "classes":
            dataset.dataset.classes,

        "image_size":
            IMAGE_SIZE
    },
    model_path
)

print()
print("=" * 40)
print("TRAINING COMPLETE")
print("=" * 40)

print(
    f"Best validation accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print("Model saved to:")

print(model_path)
