import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist import PneumoniaMNIST

DATA_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\chest_data"
MODEL_PATH = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\models\chest_pneumonia_resnet18.pth"
IMAGE_SIZE = 128
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

test_dataset = PneumoniaMNIST(
    split="test",
    root=DATA_DIR,
    transform=transform,
    as_rgb=True,
    download=False
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

probabilities = []
labels_all = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        probabilities.extend(probs.cpu().tolist())
        labels_all.extend(labels.squeeze().long().tolist())

print("Threshold sweep: pneumonia is predicted when probability >= threshold")
print("Threshold | Sensitivity | Specificity | Precision | Balanced accuracy")
print("---------------------------------------------------------------")

best = None
for threshold_int in range(30, 86, 5):
    threshold = threshold_int / 100
    tp = tn = fp = fn = 0

    for probability, actual in zip(probabilities, labels_all):
        predicted = 1 if probability >= threshold else 0
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1

    sensitivity = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    precision = tp / (tp + fp) if tp + fp else 0
    balanced = (sensitivity + specificity) / 2

    print(
        f"{threshold:8.2f} | {sensitivity * 100:10.2f}% | "
        f"{specificity * 100:10.2f}% | {precision * 100:9.2f}% | "
        f"{balanced * 100:16.2f}%"
    )

    if best is None or balanced > best[0]:
        best = (balanced, threshold, sensitivity, specificity, precision)

print("\nRecommended threshold by balanced accuracy:")
print(f"Threshold: {best[1]:.2f}")
print(f"Sensitivity: {best[2] * 100:.2f}%")
print(f"Specificity: {best[3] * 100:.2f}%")
print(f"Precision: {best[4] * 100:.2f}%")
print("\nThis only evaluates decision thresholds; it does not retrain or modify the model.")
