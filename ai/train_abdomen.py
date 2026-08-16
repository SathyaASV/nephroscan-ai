import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from medmnist import OrganAMNIST, INFO

DATA_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\abdomen_data"
MODEL_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\models"
MODEL_PATH = os.path.join(MODEL_DIR, "abdomen_organ_resnet18.pth")
IMAGE_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 6
MAX_TRAIN = 12000
MAX_VAL = 3000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(MODEL_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

label_names = INFO["organamnist"]["label"]
classes = [label_names[str(i)] for i in range(len(label_names))]
num_classes = len(classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Classes:", classes)

train_dataset = OrganAMNIST(
    split="train",
    root=DATA_DIR,
    transform=transform,
    as_rgb=True,
    download=False
)
val_dataset = OrganAMNIST(
    split="val",
    root=DATA_DIR,
    transform=transform,
    as_rgb=True,
    download=False
)

def limited_indices(length, maximum):
    indices = list(range(length))
    random.Random(SEED).shuffle(indices)
    return indices[:min(length, maximum)]

train_dataset = Subset(train_dataset, limited_indices(len(train_dataset), MAX_TRAIN))
val_dataset = Subset(val_dataset, limited_indices(len(val_dataset), MAX_VAL))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

best_accuracy = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_accuracy * 100:.2f}% | "
        f"Validation accuracy: {val_accuracy * 100:.2f}%"
    )

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "image_size": IMAGE_SIZE,
            "best_validation_accuracy": best_accuracy
        }, MODEL_PATH)
        print("Saved best model to:", MODEL_PATH)

print("\n========================================")
print("ABDOMEN ORGAN MODEL TRAINING COMPLETE")
print("========================================")
print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")
print("Model saved to:")
print(MODEL_PATH)
print("Classes:", classes)
print("This is an educational prototype, not clinical performance.")
