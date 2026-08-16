import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DATA_ROOT = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\heart_data\cardiomegaly_set\cardiomegaly_set"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
MODEL_OUTPUT = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\models\heart_cardiomegaly_resnet18_improved.pth"

IMAGE_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 8
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3

os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
classes = train_dataset.classes
if classes != test_dataset.classes:
    raise RuntimeError(f"Train/test classes differ: {classes} vs {test_dataset.classes}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Classes on disk:", classes)
print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))
print("Using device:", device)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for parameter in model.parameters():
    parameter.requires_grad = False
for parameter in model.layer4.parameters():
    parameter.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

class_counts = torch.bincount(torch.tensor(train_dataset.targets), minlength=len(classes)).float()
class_weights = (class_counts.sum() / (len(classes) * class_counts)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": BACKBONE_LR},
    {"params": model.fc.parameters(), "lr": HEAD_LR}
], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_accuracy = 0.0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    train_accuracy = 100.0 * correct / max(total, 1)

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_correct += (outputs.argmax(1) == labels).sum().item()
            test_total += labels.size(0)

    test_accuracy = 100.0 * test_correct / max(test_total, 1)
    average_loss = total_loss / max(total, 1)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {average_loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | Test Accuracy: {test_accuracy:.2f}%")

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "image_size": IMAGE_SIZE,
            "modality": "heart_cardiomegaly",
            "best_test_accuracy": best_accuracy
        }, MODEL_OUTPUT)
        print("Saved improved best model to:", MODEL_OUTPUT)

print("========================================")
print("IMPROVED HEART CARDIOMEGALY TRAINING COMPLETE")
print("========================================")
print(f"Best test accuracy: {best_accuracy:.2f}%")
print("Model saved to:", MODEL_OUTPUT)
print("Classes:", classes)
print("This is an educational prototype, not clinical performance.")
