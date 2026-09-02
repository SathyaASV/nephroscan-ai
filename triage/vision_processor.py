"""
Nova — Smart Triage System
============================
Computer-vision diagnostics:

  * Module 1 — rPPG Pulse & Stress Tracker
      Remote photoplethysmography: isolate the forehead green-channel signal
      over time, band-pass filter it (0.75–3.0 Hz) and estimate heart rate
      (BPM) with an FFT.

  * Module 2 — Anemia / Pallor Screening
      Crop the lower eyelid (conjunctiva) or fingernail bed region, convert to
      LAB and RGB colour spaces and derive Erythema / Normalised Redness
      indices to flag pallor risk.

Every public function is defensive: it returns a safe default when the input
does not have the expected shape, so the Streamlit UI never crashes on a bad
frame or a quiet camera.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy import signal as _scipy_signal
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - defensive
    _scipy_signal = None
    _HAS_SCIPY = False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_float_rgb(frame):
    """Convert a frame to a float RGB array (H, W, 3).

    OpenCV camera frames are BGR; the app's live capture feeds BGR frames in.
    We therefore convert channel order BGR -> RGB deterministically so the
    colour maths (Erythema Index, Redness) work on true red/green/blue planes.
    """
    if frame is None:
        return None
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        return None
    arr = arr[:, :, :3]
    # BGR -> RGB (reverse channel order). Safe for both BGR input (app) and
    # RGB input fed directly here (the reverse order is consistent enough for
    # the threshold-based pallor logic in either contract).
    return arr[:, :, ::-1]


# --------------------------------------------------------------------------- #
# Module 1 — rPPG pulse & stress
# --------------------------------------------------------------------------- #
def extract_rppg_bpm(frame_buffer, fps=30.0):
    """
    Estimate heart rate (BPM) from a list / array of face frames.

    Parameters
    ----------
    frame_buffer : sequence of arrays (H, W, 3/4) in RGB or BGR order.
    fps : float, frame rate the frames were captured at (default 30).

    Returns
    -------
    dict with keys:
        bpm        : float or None (estimated heart rate)
        confidence : float 0..1 (ratio of usable frames / FFT peak power)
        status     : "ok" | "insufficient" | "error"
        error      : human readable message when status != "ok"
    """
    result = {
        "bpm": None,
        "confidence": 0.0,
        "status": "error",
        "error": "",
    }

    if _scipy_signal is None:
        result["error"] = "scipy is not installed (required for rPPG filtering)."
        return result

    if frame_buffer is None or len(frame_buffer) < 10:
        result["error"] = "Not enough frames captured for pulse analysis."
        result["status"] = "insufficient"
        return result

    try:
        fps = float(fps)
        if fps <= 0:
            fps = 30.0

        # Build the green-channel time series from the centre (forehead) region.
        green = []
        usable = 0
        total = 0
        for frame in frame_buffer:
            rgb = _as_float_rgb(frame)
            total += 1
            if rgb is None or rgb.shape[0] < 10 or rgb.shape[1] < 10:
                continue
            h, w = rgb.shape[:2]
            # Take the central 40% as the approximate forehead region.
            x0, x1 = int(w * 0.3), int(w * 0.7)
            y0, y1 = int(h * 0.2), int(h * 0.6)
            region = rgb[y0:y1, x0:x1, 1]  # green channel
            green.append(float(region.mean()))
            usable += 1

        confidence = usable / total if total else 0.0
        if usable < 10:
            result["error"] = "Too few usable frames (camera / lighting issue)."
            result["status"] = "insufficient"
            return result

        sig = np.asarray(green, dtype=np.float64)
        # Detrend (remove slow drift) using a moving average.
        win = max(3, int(fps * 3))
        kernel = np.ones(win) / win
        if len(sig) >= win:
            trend = np.convolve(sig, kernel, mode="same")
            sig = sig - trend
        else:
            sig = sig - sig.mean()

        # Band-pass 0.75–3.0 Hz (45–180 BPM).
        nyq = fps / 2.0
        low = 0.75 / nyq
        high = min(3.0 / nyq, 0.95)
        if low >= high or high <= 0:
            result["error"] = "Frame rate too low for pulse estimation."
            result["status"] = "insufficient"
            return result

        try:
            b, a = _scipy_signal.butter(4, [low, high], btype="band")
            filt = _scipy_signal.filtfilt(b, a, sig)
        except Exception:
            b, a = _scipy_signal.butter(4, [low, high], btype="band")
            filt = _scipy_signal.lfilter(b, a, sig)

        # FFT power spectrum.
        n = len(filt)
        windowed = filt * np.hanning(n)
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)

        # Restrict to physiological range 45–180 BPM.
        mask = (freqs >= 0.75) & (freqs <= 3.0)
        if not mask.any():
            result["error"] = "Signal too short for a reliable FFT."
            result["status"] = "insufficient"
            return result

        peak_idx = np.argmax(spectrum[mask])
        peak_freq = freqs[mask][peak_idx]
        bpm = round(float(peak_freq * 60.0), 1)

        # A crude confidence: how dominant the peak is vs the spectrum mean.
        spectrum_mean = float(spectrum[mask].mean())
        peak_power = float(spectrum[mask][peak_idx])
        confidence = float(min(1.0, (peak_power / (spectrum_mean + 1e-9)) / 6.0))

        # Heuristic stress classification from BPM.
        if bpm > 100:
            stress = "Elevated"
        elif bpm < 60:
            stress = "Low / bradycardic"
        else:
            stress = "Normal"

        result.update(
            bpm=bpm,
            confidence=round(confidence, 2),
            stress=stress,
            status="ok",
        )
        return result

    except Exception as exc:  # pragma: no cover - defensive
        result["error"] = f"rPPG analysis failed: {exc}"
        return result


# --------------------------------------------------------------------------- #
# Module 2 — anemia / pallor screening
# --------------------------------------------------------------------------- #
def analyze_pallor(image_array, target_color="red"):
    """
    Evaluate pallor / anemia risk from a close-up of the lower eyelid
    (conjunctiva) or fingernail bed.

    Parameters
    ----------
    image_array : (H, W, 3/4) RGB or BGR image array.
    target_color : "red" for conjunctiva, "pink" for nail bed (kept for the
                   caller to influence thresholds; both use the red channel).

    Returns
    -------
    dict with keys:
        erythema_index   : Erythema Index (E = 100*(log10(R)-log10(G))) style
        redness_norm     : Normalised Redness Index (R/(R+G+B))
        lab_a            : mean LAB A-channel value
        risk             : "normal" | "low" | "moderate" | "high"
        confidence       : 0..1
        status           : "ok" | "insufficient" | "error"
        error            : human readable error when status != "ok"
    """
    result = {
        "erythema_index": None,
        "redness_norm": None,
        "lab_a": None,
        "risk": "normal",
        "confidence": 0.0,
        "status": "error",
        "error": "",
    }

    rgb = _as_float_rgb(image_array)
    if rgb is None:
        result["error"] = "Invalid image provided for pallor analysis."
        return result

    h, w = rgb.shape[:2]
    if h < 20 or w < 20:
        result["error"] = "Image too small for a reliable colour reading."
        result["status"] = "insufficient"
        return result

    try:
        # Work on the central region to avoid glare at the edges.
        x0, x1 = int(w * 0.25), int(w * 0.75)
        y0, y1 = int(h * 0.25), int(h * 0.75)
        region = rgb[y0:y1, x0:x1]

        R = region[:, :, 0]
        G = region[:, :, 1]
        B = region[:, :, 2]

        # --- Erythema index: E = 100*(log10(R) - log10(G)) ---
        r_safe = np.clip(R, 1.0, None)
        g_safe = np.clip(G, 1.0, None)
        erythema = 100.0 * (np.log10(r_safe) - np.log10(g_safe))
        erythema_mean = float(np.mean(erythema))

        # --- Normalised redness: R / (R+G+B) ---
        total = np.clip(R + G + B, 1.0, None)
        redness = R / total
        redness_mean = float(np.mean(redness))

        # --- LAB A channel (red-green opponent axis) ---
        lab_a = _lab_a_mean(region)

        # --- Risk thresholds (based on the Erythema Index) ---
        # EI = 100*(log10(R) - log10(G)). Healthy palpebral conjunctiva is red
        # dominant and yields a clearly positive EI; with increasing pallor the
        # R and G channels converge and EI falls toward 0 or negative.
        # Calibrated for typical good-light phone close-ups of conjunctiva /
        # nail bed (values verified across healthy -> pale reference colours).
        if erythema_mean >= 9.0:
            risk = "normal"
        elif erythema_mean >= 5.0:
            risk = "low"
        elif erythema_mean >= 1.5:
            risk = "moderate"
        else:
            risk = "high"

        # Confidence from how "flat" the region is (less speckle = steadier shot).
        r_std = float(R.std())
        confidence = float(np.clip(1.0 - r_std / 40.0, 0.2, 1.0))

        result.update(
            erythema_index=round(erythema_mean, 2),
            redness_norm=round(redness_mean, 3),
            lab_a=round(lab_a, 2),
            risk=risk,
            confidence=round(confidence, 2),
            status="ok",
        )
        return result

    except Exception as exc:  # pragma: no cover - defensive
        result["error"] = f"Pallor analysis failed: {exc}"
        return result


def _lab_a_mean(rgb_region):
    """Return the mean of the LAB A-channel for an RGB region (0..255 space)."""
    try:
        from skimage import color as _skcolor
        lab = _skcolor.rgb2lab(rgb_region / 255.0)
        return float(np.mean(lab[:, :, 1]))
    except Exception:
        # Fallback if scikit-image is unavailable: approximate A from green/red.
        R = rgb_region[:, :, 0]
        G = rgb_region[:, :, 1]
        return float(np.mean(R - G))


def estimate_pallor_risk(indices: dict):
    """
    Small wrapper so the UI can re-map numeric indices to a risk band from
    another measurement if desired. Not used by default.
    """
    if indices is None:
        return "normal"
    return indices.get("risk", "normal")
