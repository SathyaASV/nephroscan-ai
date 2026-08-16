import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

ROOT = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI"
MODEL_PATH = os.path.join(ROOT, "models", "heart_cardiomegaly_resnet18_improved.pth")
TEST_DIR = os.path.join(ROOT, "heart_data", "cardiomegaly_set", "cardiomegaly_set", "test")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
classes = checkpoint["classes"]
image_size = checkpoint.get("image_size", 128)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

true_index = dataset.class_to_idx.get("true", 1)
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        probabilities = torch.softmax(model(images.to(DEVICE)), dim=1)[:, true_index]
        all_probs.extend(probabilities.cpu().tolist())
        all_labels.extend((labels == true_index).int().tolist())

print("Classes:", dataset.classes)
print("Test images:", len(all_labels))
print("Positive class: true / cardiomegaly")
print("Threshold | Sensitivity | Specificity | Precision | Balanced accuracy")

best = None
for threshold_int in range(50, 100, 5):
    threshold = threshold_int / 100
    tp = tn = fp = fn = 0
    for probability, label in zip(all_probs, all_labels):
        predicted = probability >= threshold
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and not label:
            tn += 1
        else:
            fn += 1

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    balanced = (sensitivity + specificity) / 2
    print(f"{threshold:.2f}      | {sensitivity*100:6.2f}%     | {specificity*100:6.2f}%    | {precision*100:6.2f}%   | {balanced*100:6.2f}%")

    if best is None or balanced > best[0]:
        best = (balanced, threshold, sensitivity, specificity, precision)

print("\nRecommended threshold by balanced accuracy:")
print(f"Threshold: {best[1]:.2f}")
print(f"Sensitivity: {best[2]*100:.2f}%")
print(f"Specificity: {best[3]*100:.2f}%")
print(f"Precision: {best[4]*100:.2f}%")
print("These are held-out prototype metrics, not clinical performance.")
