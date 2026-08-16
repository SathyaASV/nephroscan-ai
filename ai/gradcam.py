"""
gradcam.py
Lightweight Grad-CAM for the NephroScan ResNet-18 models.

Educational prototype only. Produces an attention heatmap overlay,
NOT a lesion segmentation or diagnosis. Does not alter model
predictions in any way — this module only reads gradients/activations
that are produced as a side effect of a normal forward+backward pass.

No new third-party dependencies: uses torch, numpy, PIL only.
"""

import io
import base64

import numpy as np
import torch
from PIL import Image


def _simple_heat_colormap(norm_arr):
    """
    Map a [0,1] float array to an RGB uint8 array using a manual
    blue -> yellow -> red gradient (no matplotlib/opencv dependency).
    norm_arr: HxW float32 in [0,1]
    returns: HxWx3 uint8
    """
    r = np.clip(1.5 * norm_arr - 0.5, 0, 1)
    g = np.clip(1.5 - np.abs(2 * norm_arr - 1) * 1.5, 0, 1)
    b = np.clip(1.5 - 1.5 * norm_arr, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def generate_gradcam(model, transform, image_size, pil_image, target_layer_attr="layer3"):
    """
    Runs Grad-CAM on a single PIL image against the given model.

    Does NOT use torch.no_grad() — needs gradients. Uses a fresh
    forward/backward pass isolated from the existing prediction
    endpoints, so /predict* logic and results are unaffected.

    Returns a dict:
      {
        "predicted_index": int,
        "overlay_image": PIL.Image (RGBA overlay ready to encode),
      }

    Raises on any failure so the route can catch and return a
    graceful fallback instead of a 500 crash.
    """
    target_layer = getattr(model, target_layer_attr)

    activations = {}
    gradients = {}

    def forward_hook(_module, _inp, output):
        activations["value"] = output

    def backward_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)

        tensor = transform(pil_image).unsqueeze(0)
        tensor.requires_grad_(False)  # input itself doesn't need grad

        output = model(tensor)
        predicted_index = int(torch.argmax(output, dim=1).item())

        score = output[0, predicted_index]
        score.backward()

        acts = activations["value"][0]          # [C, H, W]
        grads = gradients["value"][0]            # [C, H, W]

        weights = grads.mean(dim=(1, 2))          # [C]
        cam = torch.relu((weights[:, None, None] * acts).sum(0))  # [H, W]

        cam = cam.detach().cpu().numpy()
        cam_max = cam.max()
        if cam_max > 1e-8:
            cam = cam / cam_max
        else:
            cam = np.zeros_like(cam)

        # Resize CAM (small feature-map resolution) up to model input size
        cam_img = Image.fromarray((cam * 255).astype(np.uint8))
        cam_img = cam_img.resize((image_size, image_size), Image.BILINEAR)
        cam_resized = np.asarray(cam_img).astype(np.float32) / 255.0

        heat_rgb = _simple_heat_colormap(cam_resized)
        heat_img = Image.fromarray(heat_rgb, mode="RGB").convert("RGBA")

        # Alpha scales with intensity so low-attention areas stay transparent
        alpha_channel = (cam_resized * 140).astype(np.uint8)  # max ~55% opacity
        heat_img.putalpha(Image.fromarray(alpha_channel, mode="L"))

        base_img = pil_image.convert("RGB").resize((image_size, image_size)).convert("RGBA")
        overlay = Image.alpha_composite(base_img, heat_img)

        return {
            "predicted_index": predicted_index,
            "overlay_image": overlay,
        }
    finally:
        h1.remove()
        h2.remove()
        model.zero_grad(set_to_none=True)


def overlay_to_base64_png(overlay_image):
    buffer = io.BytesIO()
    overlay_image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
