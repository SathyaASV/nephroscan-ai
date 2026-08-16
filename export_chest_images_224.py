import os
from PIL import Image, ImageEnhance, ImageFilter
from medmnist import PneumoniaMNIST

DATA_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\chest_data_224"
OUTPUT_DIR = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI\chest_samples_clear"

os.makedirs(os.path.join(OUTPUT_DIR, "normal"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "pneumonia"), exist_ok=True)

print("Downloading/loading the 224x224 PneumoniaMNIST test split...")
dataset = PneumoniaMNIST(
    split="test",
    root=DATA_DIR,
    download=True,
    size=224
)

normal_count = 0
pneumonia_count = 0

for image, label in zip(dataset.imgs, dataset.labels.reshape(-1)):
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

    image = Image.fromarray(image).convert("L")
    image = image.resize((768, 768), Image.Resampling.LANCZOS)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=105, threshold=2))
    image = ImageEnhance.Contrast(image).enhance(1.08)
    image.convert("RGB").save(
        os.path.join(OUTPUT_DIR, folder, filename),
        format="PNG"
    )

    if normal_count == 5 and pneumonia_count == 5:
        break

print("Clearer chest samples exported successfully.")
print("Normal images:", normal_count)
print("Pneumonia images:", pneumonia_count)
print("Open this folder:")
print(OUTPUT_DIR)
