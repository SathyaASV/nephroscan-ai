"""
Nova — Smart Triage System
============================
Signal-processing diagnostics:

  * Module 3a — Cough Acoustic Profiling (wet vs. dry)
      Analyse a cough recording's spectral content. Wet (productive) coughs
      carry more low-frequency power and a broader spectrum than sharp dry
      coughs, which are more compact and high-frequency.

  * Module 3b — Tachypnea Counter
      Count breathing rate over a 30-second window by detecting periodic
      peaks (inhalation bursts) in the audio envelope.

All functions are defensive and accept both raw `bytes` (e.g. WAV) and numpy
arrays so the Streamlit UI can feed either.
"""

from __future__ import annotations

import io

import numpy as np

try:
    import librosa

    _HAS_LIBROSA = True
except Exception:  # pragma: no cover - defensive
    librosa = None
    _HAS_LIBROSA = False

try:
    import soundfile as sf

    _HAS_SOUNDFILE = True
except Exception:  # pragma: no cover - defensive
    sf = None
    _HAS_SOUNDFILE = False

try:
    from scipy import signal as _scipy_signal

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - defensive
    _scipy_signal = None
    _HAS_SCIPY = False


# --------------------------------------------------------------------------- #
# Audio loading helpers
# --------------------------------------------------------------------------- #
def load_audio(source, sr=16000):
    """
    Load audio from bytes (WAV/MP3/FLAC/OGG) or a numpy array of samples.

    Returns a tuple (samples: np.ndarray float32 mono, sample_rate: int).
    Raises ValueError with a helpful message if the source cannot be decoded.
    """
    if source is None:
        raise ValueError("No audio data provided.")

    if isinstance(source, np.ndarray):
        y = np.asarray(source, dtype=np.float32)
        if y.ndim == 2:
            y = y.mean(axis=1)  # average channels to mono
        y = np.asarray(y, dtype=np.float32).flatten()
        return y, sr

    if isinstance(source, (bytes, bytearray)):
        if _HAS_SOUNDFILE:
            data, native_sr = sf.read(io.BytesIO(bytes(source)), dtype="float32")
            if data.ndim == 2:
                data = data.mean(axis=1)
            if native_sr != sr and native_sr > 0:
                if _HAS_LIBROSA:
                    data = librosa.resample(np.asarray(data, dtype=np.float32),
                                            orig_sr=native_sr, target_sr=sr)
                else:
                    # crude linear resample fallback
                    ratio = sr / native_sr
                    idx = np.arange(0, len(data) * ratio - 1, step=ratio, dtype=int)
                    idx = np.clip(idx, 0, len(data) - 1)
                    data = data[idx]
            return np.asarray(data, dtype=np.float32), sr
        if _HAS_LIBROSA:
            y, native_sr = librosa.load(io.BytesIO(bytes(source)), sr=sr, mono=True)
            return np.asarray(y, dtype=np.float32), sr

    raise ValueError("Unsupported audio source. Pass bytes or a numpy array.")


def _envelope(y, sr, hop=0.05):
    """Short-time RMS envelope of a mono signal."""
    valid = y[~np.isnan(y)] if np.isnan(y).any() else y
    frame_len = int(sr * hop)
    n_frames = max(1, len(valid) // frame_len)
    if n_frames * frame_len > len(valid):
        valid = valid[: n_frames * frame_len]
    shaped = valid.reshape(n_frames, frame_len)
    return np.sqrt(np.mean(shaped ** 2, axis=1) + 1e-12)


# --------------------------------------------------------------------------- #
# Module 3a — cough acoustic profiling
# --------------------------------------------------------------------------- #
def analyze_cough(source, sr=16000):
    """
    Classify a cough as 'wet' or 'dry' using spectral features.

    Returns a dict with:
        cough_type : "wet" | "dry" | "uncertain"
        risk       : risk band (drives triage)
        spectral_centroid : float (Hz) — higher for dry coughs
        wet_index          : float 0..1 probability of wetness
        duration           : seconds
        status / error
    """
    result = {
        "cough_type": "uncertain",
        "risk": "normal",
        "spectral_centroid": None,
        "wet_index": None,
        "duration": None,
        "status": "error",
        "error": "",
    }

    if not (_HAS_LIBROSA or _HAS_SOUNDFILE):
        result["error"] = "librosa/soundfile required for cough analysis."
        return result

    try:
        y, sr_used = load_audio(source, sr=sr)
        if y is None or len(y) < int(sr_used * 0.15):
            result["error"] = "Audio too short to analyse a cough."
            result["status"] = "insufficient"
            return result

        # Focus on the loudest (cough) segment.
        hop = 0.02
        env = _envelope(y, sr_used, hop=hop)
        if env.size == 0:
            result["error"] = "Could not extract audio envelope."
            return result

        peak_frame = int(np.argmax(env))
        seg_len = int(sr_used * 0.3)
        start = max(0, int(peak_frame * hop * sr_used) - seg_len // 2)
        seg = y[start: start + seg_len]

        spectral_centroid = _spectral_centroid(seg, sr_used)
        spectral_rolloff = _spectral_rolloff(seg, sr_used)

        # Wet (productive) coughs carry far more low-frequency (<500 Hz) energy
        # as a broadband "rattling" rumble, while dry coughs concentrate energy
        # in the mid/high bands. A low-frequency energy ratio is the most robust
        # discriminator, with the spectral centroid as a secondary feature.
        low_ratio, mid_ratio = _lowfreq_energy_ratio(seg, sr_used)

        # wet_index combines the low-frequency dominance with centroid evidence.
        # low_ratio dominates (0..1); a low centroid additionally nudges up.
        centroid_bias = 0.0
        if spectral_centroid is not None:
            # centroid well below ~2500 Hz favours wetness
            centroid_bias = float(np.clip(1.0 - spectral_centroid / 2500.0, 0.0, 0.4))
        wet_index = float(np.clip(low_ratio * 1.5 + centroid_bias, 0.0, 1.0))

        if wet_index >= 0.35:
            cough_type = "wet"
        elif wet_index <= 0.25:
            cough_type = "dry"
        else:
            cough_type = "uncertain"

        if cough_type == "wet":
            risk = "moderate"
        elif cough_type == "dry":
            risk = "low"
        else:
            risk = "normal"

        result.update(
            cough_type=cough_type,
            risk=risk,
            spectral_centroid=round(spectral_centroid, 1) if spectral_centroid else None,
            wet_index=round(wet_index, 2),
            duration=round(len(y) / sr_used, 2),
            status="ok",
        )
        return result

    except Exception as exc:  # pragma: no cover - defensive
        result["error"] = f"Cough analysis failed: {exc}"
        return result


def _spectral_centroid(seg, sr):
    if _HAS_LIBROSA:
        try:
            return float(librosa.feature.spectral_centroid(y=seg, sr=sr).mean())
        except Exception:
            return None
    return _fft_centroid(seg, sr)


def _spectral_rolloff(seg, sr):
    if _HAS_LIBROSA:
        try:
            return float(librosa.feature.spectral_rolloff(y=seg, sr=sr).mean())
        except Exception:
            return None
    return None


def _lowfreq_energy_ratio(seg, sr):
    """Return (low_freq_ratio, mid_freq_ratio): fraction of spectral energy
    below 500 Hz and in the 500-2000 Hz band. Wet coughs have high low_ratio."""
    try:
        valid = seg[~np.isnan(seg)] if np.isnan(seg).any() else seg
        if len(valid) == 0:
            return 0.5, 0.5
        spectrum = np.abs(np.fft.rfft(valid))
        freqs = np.fft.rfftfreq(len(valid), d=1.0 / sr)
        total = float(spectrum.sum())
        if total <= 0:
            return 0.5, 0.5
        low = float(spectrum[freqs < 500.0].sum())
        mid = float(spectrum[(freqs >= 500.0) & (freqs < 2000.0)].sum())
        return low / total, mid / total
    except Exception:
        return 0.5, 0.5


def _fft_centroid(seg, sr):
    try:
        spectrum = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
        total = spectrum.sum()
        if total <= 0:
            return None
        return float((spectrum * freqs).sum() / total)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Module 3b — tachypnea counter
# --------------------------------------------------------------------------- #
def count_breaths(source, sr=16000, window_seconds=30):
    """
    Count breaths over `window_seconds` by peak detection on the RMS envelope.

    Returns a dict with:
        breath_count   : int
        breaths_per_min : float
        risk            : risk band (uses adult threshold by default)
        status / error
    """
    result = {
        "breath_count": None,
        "breaths_per_min": None,
        "risk": "normal",
        "duration": None,
        "status": "error",
        "error": "",
    }

    if not (_HAS_LIBROSA or _HAS_SOUNDFILE):
        result["error"] = "librosa/soundfile required for breath counting."
        return result

    try:
        y, sr_used = load_audio(source, sr=sr)
        if y is None or len(y) < sr_used * 0.1:
            result["error"] = "Audio too short to count breaths."
            result["status"] = "insufficient"
            return result

        hop = 0.08  # 80 ms frames -> good for breath cadence ~ up to ~2-3 Hz
        env = _envelope(y, sr_used, hop=hop)
        if env.size < 5:
            result["error"] = "Not enough signal for breath counting."
            return result

        # Normalise and low-pass the envelope to smooth micro-variation.
        env = env - env.min()
        peak = env.max()
        if peak <= 0:
            result["error"] = "No audible breathing detected."
            result["status"] = "insufficient"
            return result
        env = env / peak

        if _scipy_signal is not None:
            try:
                b = _scipy_signal.butter(4, 0.1, btype="low")
                env = _scipy_signal.filtfilt(b, 1, env)
            except Exception:
                pass

        min_dist = int(max(2, round(0.4 / hop)))  # >= ~0.4 s between breaths
        thr = 0.35
        # Find peaks above threshold with minimum distance.
        peaks = _find_peaks(env, min_dist=min_dist, threshold=thr)

        duration = len(y) / sr_used
        breaths = int(len(peaks))
        if duration <= 0:
            result["error"] = "Invalid duration."
            return result
        bpm = breaths * (60.0 / duration)

        # Adult tachypnea threshold ~20; child threshold higher. Use 20 base.
        if bpm >= 30:
            risk = "high"
        elif bpm >= 24:
            risk = "moderate"
        elif bpm >= 20:
            risk = "low"
        else:
            risk = "normal"

        result.update(
            breath_count=breaths,
            breaths_per_min=round(bpm, 1),
            risk=risk,
            duration=round(duration, 2),
            status="ok",
        )
        return result

    except Exception as exc:  # pragma: no cover - defensive
        result["error"] = f"Breath counting failed: {exc}"
        return result


def _find_peaks(env, min_dist, threshold):
    """Simple peak finder returning indices of valid peaks."""
    peaks = []
    n = len(env)
    i = 1
    while i < n - 1:
        if env[i] > threshold and env[i] >= env[i - 1] and env[i] >= env[i + 1]:
            peaks.append(i)
            i += min_dist
        else:
            i += 1
    return peaks
