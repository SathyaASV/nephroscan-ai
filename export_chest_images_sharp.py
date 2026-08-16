import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

DATA_FILE = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\chest_data\pneumoniamnist.npz"
OUTPUT_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\chest_samples_512"

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
        folder = "normal"
        filename = f"normal_{normal_count + 1}.png"
        normal_count += 1
    elif label == 1 and pneumonia_count < 5:
        folder = "pneumonia"
        filename = f"pneumonia_{pneumonia_count + 1}.png"
        pneumonia_count += 1
    else:
        continue

    output_path = os.path.join(OUTPUT_DIR, folder, filename)

    enhanced = Image.fromarray(image).convert("L")
    enhanced = enhanced.resize((512, 512), Image.Resampling.LANCZOS)
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.12)
    enhanced.convert("RGB").save(output_path, quality=95)

    if normal_count == 5 and pneumonia_count == 5:
        break

print("512x512 presentation images exported successfully.")
print("Normal images:", normal_count)
print("Pneumonia images:", pneumonia_count)
print("Open this folder:")
print(OUTPUT_DIR)
