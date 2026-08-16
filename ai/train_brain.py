import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# Put the downloaded dataset so this folder exists:
# C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\brain_data\Brain_Tumor_MRI_Dataset\
DATASET_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\brain_data\Brain_Tumor_MRI_Dataset\Epic and CSCR hospital Dataset"
MODEL_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\models"
MODEL_PATH = os.path.join(MODEL_DIR, "brain_mri_resnet18.pth")

IMAGE_SIZE = 96
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 1e-4

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(
        "Dataset folder not found. Expected: " + DATASET_DIR
    )

train_dir = os.path.join(DATASET_DIR, "Train")
test_dir = os.path.join(DATASET_DIR, "Test")

if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
    raise FileNotFoundError(
        "Expected train and test folders inside: " + DATASET_DIR
    )

transform = transforms.Compose([
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

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_full = datasets.ImageFolder(train_dir, transform=transform)
test_full = datasets.ImageFolder(test_dir, transform=test_transform)

train_dataset = Subset(train_full, range(min(2000, len(train_full))))
test_dataset = Subset(test_full, range(min(800, len(test_full))))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

classes = train_full.classes
if classes != test_full.classes:
    raise RuntimeError("Train and test class folders do not match.")

print("Classes:", classes)
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))

try:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
except Exception:
    print("Pretrained weights unavailable; using an uninitialized ResNet-18.")
    model = models.resnet18(weights=None)

model.fc = nn.Linear(model.fc.in_features, len(classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_accuracy = 0.0
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100.0 * correct / total

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * test_correct / test_total
    epoch_loss = running_loss / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {epoch_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_weights = copy.deepcopy(model.state_dict())

model.load_state_dict(best_weights)

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes,
    "image_size": IMAGE_SIZE,
    "model_name": "resnet18"
}, MODEL_PATH)

print("\n========================================")
print("BRAIN MRI MODEL TRAINING COMPLETE")
print("========================================")
print(f"Best test accuracy: {best_accuracy:.2f}%")
print("Model saved to:")
print(MODEL_PATH)
print("Classes:", classes)
