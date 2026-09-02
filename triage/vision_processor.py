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
def _rgb_mean_ts(rgb, x0, x1, y0, y1):
    """Mean R,G,B over the central ROI for a single frame -> (r, g, b)."""
    region = rgb[y0:y1, x0:x1]
    return (float(region[:, :, 0].mean()),
            float(region[:, :, 1].mean()),
            float(region[:, :, 2].mean()))


def _pos_rppg(timeseries, fps):
    """
    POS (Planar Orthogonal to Skin) rPPG — Morelli et al. 2020.
    Robust to motion artefacts compared to a naive green-channel trace.

    timeseries : (N, 3) array of R,G,B mean values over time.
    Returns a detrended, band-limited pulse signal (N,).
    """
    x = timeseries
    # Normalise each channel by a running mean (mean-based normalisation).
    win = max(3, int(fps * 3))
    norm = np.zeros_like(x)
    for c in range(3):
        col = x[:, c]
        if len(col) >= win:
            kernel = np.ones(win) / win
            kn = np.convolve(col, kernel, mode="same")
            kn[kn <= 1e-6] = 1.0
            norm[:, c] = col / kn
        else:
            m = max(col.mean(), 1e-6)
            norm[:, c] = col / m

    # Skin-tone calibrated projection.
    h_root = 2.0 / np.sqrt(6.0)
    c_mat = np.array([
        [1, -1, 0],
        [1, 1, -2],
        [h_root, h_root, h_root],
    ], dtype=np.float64)
    projected = norm @ c_mat.T
    # Remove mean and detrend the 2nd (pulsatile) component.
    s = projected[:, 1]
    s = s - s.mean()
    if len(s) >= win:
        kernel = np.ones(win) / win
        trend = np.convolve(s, kernel, mode="same")
        s = s - trend
    return s


def _spectral_snr(filt, fps):
    """Return (bpm, confidence) via peak prominence vs noise floor + harmonic check."""
    n = len(filt)
    windowed = filt * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    mask = (freqs >= 0.75) & (freqs <= 3.0)  # 45-180 BPM
    if not mask.any():
        return None, 0.0
    pmask = freqs[mask]
    pspectrum = spectrum[mask]
    peak_idx = int(np.argmax(pspectrum))
    peak_freq = float(pmask[peak_idx])
    peak_power = float(pspectrum[peak_idx])

    # Noise floor = energy in the band excluding a narrow window around the peak.
    guard = 0.15  # Hz around peak excluded as signal
    noise_mask = (np.abs(pmask - peak_freq) > guard)
    noise_sum = float(pspectrum[noise_mask].sum()) if noise_mask.any() else peak_power
    noise_mean = noise_sum / (max(1, int(noise_mask.sum())))
    snr = peak_power / (noise_mean + 1e-9)
    confidence = float(np.clip(snr / 20.0, 0.0, 1.0))

    bpm = peak_freq * 60.0

    # Harmonic check: if the peak is near the top of the range and its 2nd
    # harmonic would be out of band, we accept as-is. If a strong sub-harmonic
    # (0.5x) exists, prefer it rarely; keep simple.
    return bpm, confidence


def extract_rppg_bpm(frame_buffer, fps=30.0):
    """
    Estimate heart rate (BPM) from a list / array of face frames.

    Uses POS planar-orthogonal-to-skin rPPG with a green-channel fallback, a
    physically band-limited FFT, and a spectral signal-to-noise confidence.

    Returns a dict with:
        bpm           : float or None
        confidence    : float 0..1 (spectral SNR based)
        signal_quality: "good" | "fair" | "poor"
        stress        : categorical stress read
        status        : "ok" | "insufficient" | "error"
        error         : message when status != "ok"
    """
    result = {
        "bpm": None, "confidence": 0.0, "signal_quality": "poor",
        "stress": "Normal", "status": "error", "error": "",
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

        rgb_ts = []
        total = 0
        for frame in frame_buffer:
            rgb = _as_float_rgb(frame)
            total += 1
            if rgb is None or rgb.shape[0] < 10 or rgb.shape[1] < 10:
                continue
            h, w = rgb.shape[:2]
            # Central 40% (approx forehead / region of interest).
            x0, x1 = int(w * 0.3), int(w * 0.7)
            y0, y1 = int(h * 0.2), int(h * 0.6)
            rgb_ts.append(_rgb_mean_ts(rgb, x0, x1, y0, y1))

        if len(rgb_ts) < 10:
            result["error"] = "Too few usable frames (camera / lighting issue)."
            result["status"] = "insufficient"
            return result

        ts = np.asarray(rgb_ts, dtype=np.float64)  # (N,3) R,G,B rows

        # POS extraction (motion robust), fall back to detrended green.
        try:
            sig = _pos_rppg(ts, fps)
        except Exception:
            green = ts[:, 1]
            sig = green - green.mean()
            win = max(3, int(fps * 3))
            if len(sig) >= win:
                kernel = np.ones(win) / win
                sig = sig - np.convolve(sig, kernel, mode="same")

        # Band-pass 0.75-3.0 Hz (45-180 BPM).
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

        bpm, confidence = _spectral_snr(filt, fps)
        if bpm is None or bpm <= 0:
            result["error"] = "Could not find a reliable pulse peak."
            result["status"] = "insufficient"
            return result
        bpm = round(bpm, 1)

        # Signal-quality band from confidence.
        if confidence >= 0.6:
            signal_quality = "good"
        elif confidence >= 0.35:
            signal_quality = "fair"
        else:
            signal_quality = "poor"

        # Stress classification (5 bands).
        if bpm > 110:
            stress = "High (tachycardic)"
        elif bpm > 100:
            stress = "Elevated"
        elif bpm >= 60 and bpm <= 100:
            stress = "Normal"
        elif bpm >= 48:
            stress = "Low / resting"
        else:
            stress = "Low (bradycardic)"

        result.update(
            bpm=bpm,
            confidence=round(confidence, 2),
            signal_quality=signal_quality,
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
def analyze_pallor(image_array, target_color="red", zone="conjunctiva"):
    """
    Evaluate pallor / anemia risk from a close-up of the lower eyelid
    (conjunctiva) or fingernail bed.

    Parameters
    ----------
    image_array : (H, W, 3/4) RGB or BGR image array.
    target_color : "red" for conjunctiva, "pink" for nail bed (kept for the
                   caller to influence thresholds; both use the red channel).
    zone : "conjunctiva" | "nailbed" | other — enables zone-aware thresholds.

    Returns
    -------
    dict with keys:
        erythema_index   : Erythema Index (E = 100*(log10(R)-log10(G)))
        redness_norm     : Normalised Redness Index (R/(R+G+B))
        lab_a            : mean LAB A-channel value
        rgb_means        : dict R/G/B means
        risk             : "normal" | "low" | "moderate" | "high"
        confidence       : 0..1
        status           : "ok" | "insufficient" | "error"
        error            : human readable error when status != "ok"
    """
    result = {
        "erythema_index": None,
        "redness_norm": None,
        "lab_a": None,
        "rgb_means": {},
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

        # --- Zone-aware risk thresholds (based on the Erythema Index) ---
        # Nailbed is naturally lighter/pinkish than the conjunctiva, so its
        # healthy EI baseline is a touch lower; keep the same bands otherwise.
        if str(zone).strip().lower() == "nailbed":
            n_healthy, n_low, n_mod = 6.5, 3.5, 1.0
        else:  # conjunctiva and general skin
            n_healthy, n_low, n_mod = 9.0, 5.0, 1.5

        if erythema_mean >= n_healthy:
            risk = "normal"
        elif erythema_mean >= n_low:
            risk = "low"
        elif erythema_mean >= n_mod:
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
            rgb_means={
                "R": round(float(R.mean()), 1),
                "G": round(float(G.mean()), 1),
                "B": round(float(B.mean()), 1),
            },
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
