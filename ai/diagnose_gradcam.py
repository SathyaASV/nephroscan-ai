"""
Grad-CAM diagnostics for NephroScan AI.
Inspects ai_server.py /explain and ai/gradcam.py internals.
Runs one Brain sample and one Heart sample.
Does NOT modify any code — purely diagnostic.
"""
import io, os, sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = r"C:\Users\Panchami.A\OneDrive\Desktop\NephroScan-AI"
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# ---- Replicate model loading from ai_server.py ----
def load_checkpoint(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 128)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, classes, image_size

# ---- Manually instrument Grad-CAM to capture internal tensors ----
def gradcam_diagnostics(model, transform, image_size, pil_image, target_layer_attr="layer4"):
    target_layer = getattr(model, target_layer_attr)
    activations = {}
    gradients = {}

    def fwd_hook(_module, _inp, output):
        activations["value"] = output
    def bwd_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad(set_to_none=True)
        tensor = transform(pil_image).unsqueeze(0)
        output = model(tensor)
        predicted_index = int(torch.argmax(output, dim=1).item())
        score = output[0, predicted_index]
        score.backward()

        acts = activations["value"][0]   # [C, H, W]
        grads = gradients["value"][0]      # [C, H, W]

        act_shape = tuple(acts.shape)
        grad_shape = tuple(grads.shape)

        weights = grads.mean(dim=(1, 2))          # [C]
        cam = torch.relu((weights[:, None, None] * acts).sum(0))  # [H, W]
        cam = cam.detach().cpu().numpy()
        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max
        else:
            cam = np.zeros_like(cam)

        # Resize CAM to image_size
        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_img = cam_img.resize((image_size, image_size), Image.BILINEAR)
        cam_resized = np.asarray(cam_img).astype(np.float32) / 255.0

        # Colormap + alpha
        r = np.clip(1.5 * cam_resized - 0.5, 0, 1)
        g = np.clip(1.5 - np.abs(2 * cam_resized - 1) * 1.5, 0, 1)
        b = np.clip(1.5 - 1.5 * cam_resized, 0, 1)
        heat_rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        heat_img = Image.fromarray(heat_rgb, mode="RGB").convert("RGBA")
        alpha_channel = (cam_resized * 140).astype(np.uint8)
        heat_img.putalpha(Image.fromarray(alpha_channel, mode="L"))

        base_img = pil_image.convert("RGB").resize((image_size, image_size)).convert("RGBA")
        overlay = Image.alpha_composite(base_img, heat_img)

        return {
            "predicted_index": predicted_index,
            "act_shape": act_shape,
            "grad_shape": grad_shape,
            "cam_raw": cam,            # at layer4 resolution
            "cam_resized": cam_resized, # at image_size resolution
            "alpha_channel": alpha_channel,
            "heat_rgb": heat_rgb,
            "overlay": overlay,
            "cam_img_at_layer4": cam_img,  # PIL of raw CAM resized (grayscale)
            "base_img_resized": base_img,
            "heat_img_rgba": heat_img,
        }
    finally:
        h1.remove()
        h2.remove()
        model.zero_grad(set_to_none=True)


def overlay_to_base64_png(overlay_image):
    buffer = io.BytesIO()
    overlay_image.convert("RGB").save(buffer, format="PNG")
    import base64
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ================================================================
# BRAIN sample
# ================================================================
brain_model, brain_classes, brain_image_size = load_checkpoint(
    os.path.join(MODELS_DIR, "brain_mri_resnet18.pth")
)
brain_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((brain_image_size, brain_image_size)),
    transforms.ToTensor(),
    NORMALIZE,
])

brain_sample_path = os.path.join(
    PROJECT_ROOT,
    "brain_data", "Brain_Tumor_MRI_Dataset",
    "Epic and CSCR hospital Dataset", "Train", "glioma", "gg (334).jpg"
)
brain_pil = Image.open(brain_sample_path).convert("RGB")

brain_result = gradcam_diagnostics(
    brain_model, brain_transform, brain_image_size, brain_pil
)

# ================================================================
# HEART sample
# ================================================================
heart_model, heart_classes, heart_image_size = load_checkpoint(
    os.path.join(MODELS_DIR, "heart_cardiomegaly_resnet18_improved.pth")
)
heart_transform = transforms.Compose([
    transforms.Resize((heart_image_size, heart_image_size)),
    transforms.ToTensor(),
    NORMALIZE,
])

heart_sample_path = os.path.join(
    PROJECT_ROOT, "abdomen_samples_clear", "heart", "heart_1_clear.png"
)
heart_pil = Image.open(heart_sample_path).convert("RGB")

heart_result = gradcam_diagnostics(
    heart_model, heart_transform, heart_image_size, heart_pil
)

# ================================================================
# REPORT
# ================================================================
def report(section, classes, image_size, transform, result, pil_image):
    print(f"\n{'='*72}")
    print(f"  {section}")
    print(f"{'='*72}")

    print(f"\n1. Model input image size: {image_size}x{image_size}")
    print(f"   Normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
    print(f"   Transform pipeline:")
    for t in transform.transforms:
        print(f"     - {t}")

    print(f"\n2. Actual target layer: model.layer4 (ResNet-18 final Bottleneck block)")
    print(f"   (gradient target = argmax class logit -> backward from that score)")

    print(f"\n3. Predicted class index: {result['predicted_index']}")
    print(f"   Predicted class: {classes[result['predicted_index']]}")
    print(f"   All classes: {classes}")

    print(f"\n4. Activation tensor shape: {result['act_shape']}  [C, H, W]")
    print(f"   Gradient tensor shape:  {result['grad_shape']}  [C, H, W]")

    cam_r = result["cam_resized"]
    pct_above = (cam_r > 0.5).sum() / cam_r.size * 100
    print(f"\n5. Heatmap (CAM) statistics at image_size resolution:")
    print(f"   min:    {cam_r.min():.6f}")
    print(f"   max:    {cam_r.max():.6f}")
    print(f"   mean:   {cam_r.mean():.6f}")
    print(f"   % pixels > 0.5: {pct_above:.2f}%")

    print(f"\n6. Original image dimensions: {pil_image.size} (WxH)")
    print(f"   Heatmap (resized CAM) dimensions: {cam_r.shape[1]}x{cam_r.shape[0]}  (HxW)")
    print(f"   (CAM at layer4 resolution: {result['cam_raw'].shape[1]}x{result['cam_raw'].shape[0]})")

    print(f"\n7. Returned overlay is ALREADY COMPOSITED with the original image:")
    print(f"   - base_img resized -> {image_size}x{image_size} RGBA")
    print(f"   - heatmap alpha-blended via Image.alpha_composite(base_img, heat_img)")
    print(f"   - alpha cap: cam_resized * 140 (max ~54.9% opacity at peak)")
    print(f"   - NOT transparent-only; original pixels are visible underneath")

    print(f"\n8. Aspect ratio preservation:")
    orig_w, orig_h = pil_image.size
    print(f"   Original: {orig_w}x{orig_h} -> {orig_w/orig_h:.3f} aspect")
    print(f"   Model input: {image_size}x{image_size} (square, aspect-ratio STRETCHED)")
    print(f"   Overlay output: {image_size}x{image_size} (square)")
    print(f"   -> Original aspect ratio is NOT preserved (forced to square)")

    return cam_r

brain_cam_stats = report("BRAIN MRI (glioma sample gg (334).jpg)", brain_classes, brain_image_size, brain_transform, brain_result, brain_pil)
heart_cam_stats = report("HEART (heart_1_clear.png)", heart_classes, heart_image_size, heart_transform, heart_result, heart_pil)

# ================================================================
# DIAGNOSTIC IMAGES
# ================================================================
diag_dir = os.path.join(PROJECT_ROOT, "diagnostics")
os.makedirs(diag_dir, exist_ok=True)

def save_diagnostics(prefix, pil_image, result, image_size):
    # 1. Original image as-is
    pil_image.save(os.path.join(diag_dir, f"{prefix}_original.png"))

    # 2. Base image resized to model input (showing how it's stretched)
    result["base_img_resized"].save(os.path.join(diag_dir, f"{prefix}_base_resized.png"))

    # 3. Raw CAM at layer4 resolution (grayscale, upscaled to image_size for visibility)
    cam_pil = Image.fromarray((result["cam_raw"] * 255).astype(np.uint8))
    cam_pil = cam_pil.resize((image_size * 4, image_size * 4), Image.NEAREST)  # pixelated view
    cam_pil.save(os.path.join(diag_dir, f"{prefix}_cam_raw_layer4_upscaled.png"))

    # 4. CAM resized to image_size (the actual heatmap intensity map)
    cam_resized_pil = Image.fromarray((result["cam_resized"] * 255).astype(np.uint8))
    cam_resized_pil.save(os.path.join(diag_dir, f"{prefix}_cam_resized.png"))

    # 5. Heatmap RGB (colored, before alpha)
    Image.fromarray(result["heat_rgb"]).save(os.path.join(diag_dir, f"{prefix}_heatmap_rgb.png"))

    # 6. Alpha channel (single channel)
    Image.fromarray(result["alpha_channel"], mode="L").save(os.path.join(diag_dir, f"{prefix}_alpha_channel.png"))

    # 7. Final composited overlay (as the API returns it)
    result["overlay"].save(os.path.join(diag_dir, f"{prefix}_overlay_final.png"))

    # 8. Side-by-side comparison: original | heatmap-only | overlay
    w = image_size
    comparison = Image.new("RGB", (w * 3, w), color=(50, 50, 50))
    comparison.paste(result["base_img_resized"].convert("RGB"), (0, 0))
    comparison.paste(Image.fromarray(result["heat_rgb"]), (w, 0))
    comparison.paste(result["overlay"].convert("RGB"), (w * 2, 0))
    comparison.save(os.path.join(diag_dir, f"{prefix}_comparison.png"))

    print(f"\n   Saved diagnostics to: {diag_dir}/")

save_diagnostics("brain", brain_pil, brain_result, brain_image_size)
save_diagnostics("heart", heart_pil, heart_result, heart_image_size)

print(f"\n{'='*72}")
print("  DIAGNOSTIC SUMMARY")
print(f"{'='*72}")

print(f"\nBRAIN (glioma):")
print(f"  layer4 CAM shape: {brain_result['cam_raw'].shape}")
print(f"  CAM min={brain_result['cam_resized'].min():.4f}  max={brain_result['cam_resized'].max():.4f}  mean={brain_result['cam_resized'].mean():.4f}")
brain_pct = (brain_result['cam_resized'] > 0.5).sum() / brain_result['cam_resized'].size * 100
print(f"  % pixels > 0.5: {brain_pct:.2f}%")
print(f"  Unique CAM values at layer4: {len(np.unique(brain_result['cam_raw']))} distinct values")
print(f"  Unique CAM values after resize: {len(np.unique(brain_result['cam_resized']))} distinct values")

print(f"\nHEART (cardiomegaly):")
print(f"  layer4 CAM shape: {heart_result['cam_raw'].shape}")
print(f"  CAM min={heart_result['cam_resized'].min():.4f}  max={heart_result['cam_resized'].max():.4f}  mean={heart_result['cam_resized'].mean():.4f}")
heart_pct = (heart_result['cam_resized'] > 0.5).sum() / heart_result['cam_resized'].size * 100
print(f"  % pixels > 0.5: {heart_pct:.2f}%")
print(f"  Unique CAM values at layer4: {len(np.unique(heart_result['cam_raw']))} distinct values")
print(f"  Unique CAM values after resize: {len(np.unique(heart_result['cam_resized']))} distinct values")

print(f"\n{'='*72}")
print("  END DIAGNOSTICS — no code changes were made")
print(f"{'='*72}")
