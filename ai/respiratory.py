"""
respiratory.py -- Nova respiratory triage inference (folded into NephroScan/AI).

Deterministic port of the Nova smart-stethoscope respiratory pipeline:
MobileNetV3-Small, 4 classes (Normal / Crackles / Wheezes / Both), fed with a
5-second log-mel spectrogram rendered as an RGB image.

Used by the Flask server's POST /api/respiratory endpoint. Educational
prototype only; not a substitute for clinical diagnosis.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

logger = logging.getLogger("nova.respiratory")

# ---- Audio / feature constants (mirror config.py of the Nova app) ----
SAMPLE_RATE = 16_000
DURATION = 5.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)          # 80,000
BANDPASS_LOW_HZ = 100.0
BANDPASS_HIGH_HZ = 2_000.0
BANDPASS_ORDER = 5
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
IMAGE_SIZE = (224, 224)

# ---- Class mappings ----
RESP_CLASS_LABELS = ["Normal", "Crackles", "Wheezes", "Both"]

STATUS_COLORS = {
    "Normal": "#059669",
    "Wheezes": "#ca8a04",
    "Crackles": "#dc2626",
    "Both": "#991b1b",
}

RISK_META = {
    "Normal": {"level": "LOW", "label": "Low Risk", "action": "Routine follow-up"},
    "Crackles": {"level": "HIGH", "label": "High Risk", "action": "Immediate referral"},
    "Wheezes": {"level": "MODERATE", "label": "Moderate Risk", "action": "Bronchodilator + follow-up"},
    "Both": {"level": "CRITICAL", "label": "Critical Risk", "action": "Emergency referral"},
}

CLINICAL_ADVICE = {
    "Normal": "Clear breath sounds. Patient is stable. Continue routine follow-up.",
    "Crackles": "Discontinuous clicking sounds detected. Potential fluid or pneumonia risk. Refer to a medical officer immediately.",
    "Wheezes": "Continuous whistling sounds detected. Potential airway constriction or asthma. Administer prescribed bronchodilator.",
    "Both": "Severe crackles and wheezes detected. Critical respiratory risk. High-priority referral to specialist required.",
}

MODEL_FILENAME = "respiratory_classifier.pth"

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _normalize_state_dict(raw):
    """Unwrap common checkpoint formats into a plain state dict."""
    if not isinstance(raw, dict):
        return {}
    if "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
        sd = raw["model_state_dict"]
    elif "state_dict" in raw and isinstance(raw["state_dict"], dict):
        sd = raw["state_dict"]
    elif "model" in raw and isinstance(raw["model"], dict):
        sd = raw["model"]
    else:
        if raw and all(isinstance(v, torch.Tensor) for v in raw.values()):
            sd = raw
        else:
            return {}
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


def _remap_unprefixed_keys(state_dict):
    """Prefix bare feature/classifier keys with ``backbone.`` (legacy checkpoints)."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith("backbone.") or k.startswith("fc.") or k.startswith("conv"):
            out[k] = v
        elif k.startswith("features.") or k.startswith("avgpool."):
            out["backbone." + k] = v
        elif k.startswith("classifier."):
            out["backbone." + k] = v
        else:
            out[k] = v
    return out


class RespiratoryClassifier(nn.Module):
    """MobileNetV3-Small with a custom 4-class head (matches trained weights)."""

    def __init__(self, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        from torchvision import models as tv_models
        # Offline-first: random init; the checkpoint restores real weights.
        try:
            self.backbone = tv_models.mobilenet_v3_small(weights=None)
        except Exception:  # pragma: no cover - fallback
            self.backbone = tv_models.mobilenet_v3_small(weights=None)
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class RespiratoryEngine:
    """Loads the respiratory checkpoint and runs triage inference."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.using_fallback = False

    def load(self) -> bool:
        try:
            model = RespiratoryClassifier()
            try:
                ckpt = torch.load(self.model_path, map_location=_DEVICE, weights_only=True)
            except Exception:
                ckpt = torch.load(self.model_path, map_location=_DEVICE, weights_only=False)
            state_dict = _normalize_state_dict(ckpt)
            if not state_dict:
                logger.error("Unrecognised checkpoint: %s", self.model_path)
                return False
            state_dict = _remap_unprefixed_keys(state_dict)
            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError:
                inc = model.load_state_dict(state_dict, strict=False)
                if inc.missing_keys or inc.unexpected_keys:
                    logger.warning(
                        "Respiratory tolerant load: %d missing, %d unexpected",
                        len(inc.missing_keys), len(inc.unexpected_keys),
                    )
            model.to(_DEVICE)
            model.eval()
            self.model = model
            logger.info("Respiratory model loaded from %s", self.model_path)
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to load respiratory model: %s", e, exc_info=True)
            self.using_fallback = True
            return False

    def _preprocess(self, wav_bytes):
        """Full audio pipeline -> (1,3,224,224) normalized RGB tensor."""
        import librosa
        from scipy import signal as scipy_signal

        waveform, sr = librosa.load(io.BytesIO(wav_bytes), sr=SAMPLE_RATE, mono=True)
        waveform = waveform.astype(np.float32)

        # trim / zero-pad to exactly 5 s worth of samples
        if len(waveform) > NUM_SAMPLES:
            waveform = waveform[:NUM_SAMPLES]
        elif len(waveform) < NUM_SAMPLES:
            waveform = np.pad(waveform, (0, NUM_SAMPLES - len(waveform)), mode="constant")

        # Butterworth bandpass 100-2000 Hz
        nyq = 0.5 * SAMPLE_RATE
        b, a = scipy_signal.butter(
            BANDPASS_ORDER, [BANDPASS_LOW_HZ / nyq, BANDPASS_HIGH_HZ / nyq], btype="band"
        )
        waveform = scipy_signal.filtfilt(b, a, waveform).astype(np.float32)

        # log-mel spectrogram (dB)
        mel = librosa.feature.melspectrogram(
            y=waveform, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
            hop_length=HOP_LENGTH, power=2.0, fmin=0.0, fmax=SAMPLE_RATE / 2,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # normalize to [0,1], then to an RGB image, resize to 224x224
        spec_norm = np.clip((log_mel - (-80.0)) / (0.0 - (-80.0)), 0.0, 1.0)
        spec_img = Image.fromarray((spec_norm * 255).astype(np.uint8))
        spec_img = spec_img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        spec_rgb = np.stack([np.array(spec_img)] * 3, axis=-1) / 255.0

        tensor = torch.from_numpy(spec_rgb.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor.to(_DEVICE)

    def predict(self, wav_bytes: bytes) -> dict:
        if self.model is None:
            if not self.load():
                return {"error": "Respiratory model not available"}

        with torch.no_grad():
            x = self._preprocess(wav_bytes)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            label = RESP_CLASS_LABELS[pred_idx]
            confidence = float(probs[pred_idx] * 100)

        result = {
            "label": label,
            "confidence_pct": round(confidence, 1),
            "color": STATUS_COLORS.get(label, "#64748b"),
            "risk_level": RISK_META.get(label, {}).get("level", "LOW"),
            "risk_label": RISK_META.get(label, {}).get("label", "Low Risk"),
            "action": RISK_META.get(label, {}).get("action", "Follow-up"),
            "advice": CLINICAL_ADVICE.get(label, "Consult physician"),
            "all_confidences": {
                RESP_CLASS_LABELS[i]: float(probs[i] * 100)
                for i in range(len(RESP_CLASS_LABELS))
            },
            "using_fallback_weights": self.using_fallback,
        }
        return result


# Module-level singleton (loaded lazily by the Flask app).
_engine: RespiratoryEngine | None = None


def get_engine(model_dir) -> RespiratoryEngine:
    """Return the shared respiratory engine, loading on first use."""
    global _engine
    if _engine is None:
        from pathlib import Path
        _engine = RespiratoryEngine(Path(model_dir) / MODEL_FILENAME)
        _engine.load()
    return _engine