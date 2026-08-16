import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

DATA_FILE = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\abdomen_data\organamnist.npz"
OUTPUT_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\abdomen_samples_clear"
CLASS_NAMES = [
    "bladder", "femur-left", "femur-right", "heart",
    "kidney-left", "kidney-right", "liver", "lung-left",
    "lung-right", "pancreas", "spleen"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
for name in CLASS_NAMES:
    os.makedirs(os.path.join(OUTPUT_DIR, name), exist_ok=True)

data = np.load(DATA_FILE)
images = data["test_images"]
labels = data["test_labels"].reshape(-1)
counts = {name: 0 for name in CLASS_NAMES}

for image, label in zip(images, labels):
    index = int(label)
    if index < 0 or index >= len(CLASS_NAMES):
        continue
    name = CLASS_NAMES[index]
    if counts[name] >= 3:
        continue

    gray = Image.fromarray(image).convert("L")
    clear = gray.resize((512, 512), Image.Resampling.LANCZOS)
    clear = ImageEnhance.Contrast(clear).enhance(1.35)
    clear = clear.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))
    output = os.path.join(OUTPUT_DIR, name, f"{name}_{counts[name] + 1}_clear.png")
    clear.save(output)
    counts[name] += 1

    if all(value >= 3 for value in counts.values()):
        break

print("Clear display versions exported.")
print("Open this folder:", OUTPUT_DIR)
for name in CLASS_NAMES:
    print(name, counts[name])
