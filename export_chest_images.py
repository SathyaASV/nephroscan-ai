import os
import numpy as np
from PIL import Image

DATA_FILE = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\chest_data\pneumoniamnist.npz"
OUTPUT_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\chest_samples"

os.makedirs(os.path.join(OUTPUT_DIR, "normal"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "pneumonia"), exist_ok=True)

data = np.load(DATA_FILE)
images = data["test_images"]
labels = data["test_labels"].reshape(-1)

normal_count = 0
pneumonia_count = 0

for image, label in zip(images, labels):
    label = int(label)

    if label == 0 and normal_count < 5:
        output_path = os.path.join(
            OUTPUT_DIR,
            "normal",
            f"normal_{normal_count + 1}.png"
        )
        Image.fromarray(image).convert("RGB").save(output_path)
        normal_count += 1

    elif label == 1 and pneumonia_count < 5:
        output_path = os.path.join(
            OUTPUT_DIR,
            "pneumonia",
            f"pneumonia_{pneumonia_count + 1}.png"
        )
        Image.fromarray(image).convert("RGB").save(output_path)
        pneumonia_count += 1

    if normal_count == 5 and pneumonia_count == 5:
        break

print("Chest sample images exported successfully.")
print("Normal images:", normal_count)
print("Pneumonia images:", pneumonia_count)
print("Open this folder:")
print(OUTPUT_DIR)

