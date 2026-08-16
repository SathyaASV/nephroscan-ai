import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# =========================
# PATHS
# =========================

DATASET_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\my dataset final 512x512(implemented)"
MODEL_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# SETTINGS
# =========================

IMAGE_SIZE = 160
BATCH_SIZE = 16
EPOCHS = 8

device = torch.device("cpu")

print("Using device:", device)

# =========================
# TRANSFORMS
# =========================

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD DATASET
# =========================

base_dataset = datasets.ImageFolder(DATASET_DIR)

classes = base_dataset.classes
targets = torch.tensor(base_dataset.targets)

print("Classes:", classes)
print("Total images:", len(base_dataset))

# =========================
# STRATIFIED SPLIT
# =========================

train_indices = []
val_indices = []

generator = torch.Generator().manual_seed(42)

for class_id in range(len(classes)):

    class_indices = torch.where(targets == class_id)[0]

    shuffled = class_indices[
        torch.randperm(
            len(class_indices),
            generator=generator
        )
    ]

    split = int(0.8 * len(shuffled))

    train_indices.extend(shuffled[:split].tolist())
    val_indices.extend(shuffled[split:].tolist())

train_indices = torch.tensor(train_indices)
val_indices = torch.tensor(val_indices)

print("Training images:", len(train_indices))
print("Validation images:", len(val_indices))

# =========================
# CREATE DATASETS
# =========================

train_full = datasets.ImageFolder(
    DATASET_DIR,
    transform=train_transform
)

val_full = datasets.ImageFolder(
    DATASET_DIR,
    transform=val_transform
)

train_dataset = Subset(
    train_full,
    train_indices.tolist()
)

val_dataset = Subset(
    val_full,
    val_indices.tolist()
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

# =========================
# RESNET-18
# =========================

print("Loading ResNet-18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Fine-tune the final ResNet block
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace classifier
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(classes)
)

model = model.to(device)

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)

# =========================
# TRAINING
# =========================

best_accuracy = 0.0
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):

    print()
    print("=" * 45)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 45)

    # ---------------------
    # TRAIN
    # ---------------------

    model.train()

    total = 0
    correct = 0
    running_loss = 0.0

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
            loss.item() * images.size(0)
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

    # ---------------------
    # VALIDATION
    # ---------------------

    model.eval()

    total = 0
    correct = 0
    val_loss_total = 0.0

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
                loss.item() * images.size(0)
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

    print(f"Train Loss:       {train_loss:.4f}")
    print(f"Train Accuracy:   {train_accuracy * 100:.2f}%")
    print(f"Validation Loss:  {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_weights = copy.deepcopy(
            model.state_dict()
        )

        print("★ New best model!")

# =========================
# SAVE V2
# =========================

model.load_state_dict(best_weights)

model_path = os.path.join(
    MODEL_DIR,
    "kidney_stone_resnet18_v2.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "image_size": IMAGE_SIZE
    },
    model_path
)

print()
print("=" * 45)
print("V2 TRAINING COMPLETE")
print("=" * 45)

print(
    f"Best validation accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print("New model saved to:")
print(model_path)

print()
print("IMPORTANT:")
print("The original kidney_stone_resnet18.pth was NOT changed.")