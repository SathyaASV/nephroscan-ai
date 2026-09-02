"""
Nova — Smart Triage System
============================
Central configuration for the offline-first rural healthcare diagnostics app.

Holds project constants, the supported Indian language map, anatomical scan
zones with placement guidance, and the base clinical triage advice used to
build spoken and written reports.

All modules import from here so there is a single source of truth.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Project identity
# --------------------------------------------------------------------------- #
PROJECT_NAME = "Nova"
PROJECT_TAGLINE = "Smart Triage System"
SUPPORTING_PROGRAM = "INSPIRE-MANAK Project"

# A rough default on most systems. Audio is captured at the mic's native rate
# and down-sampled to SAMPLE_RATE before analysis where needed.
SAMPLE_RATE = 16000

# When collecting rPPG frames keep this many seconds worth of signal.
RPPG_BUFFER_SECONDS = 20

# Breath counting window (seconds) used by the tachypnea counter.
BREATH_WINDOW_SECONDS = 30

# Root directory of the project (parent of this file) so offline assets can
# resolve regardless of the working directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Indian regional languages
#
# Keys are stable ISO-ish codes used throughout the app (UI, gTTS, deep
# translator). Values are the human readable names.
# --------------------------------------------------------------------------- #
LANGUAGES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "mr": "Marathi (मराठी)",
    "bn": "Bengali (বাংলা)",
    "gu": "Gujarati (ગુજરાતી)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
}

# Default UI language and the language spoken by default in the voice report.
DEFAULT_LANGUAGE = "en"

# --------------------------------------------------------------------------- #
# Anatomical scan zones with placement guidance
#
# Each entry carries the category (info shown to the worker), a friendly name,
# an info message for the Smart Placement Guide and a short advice string that
# gets logged as metadata in the exported report.
# --------------------------------------------------------------------------- #
SCAN_ZONES = {
    # --- Acoustic zones (lung auscultation / cough capture) --------------- #
    "url": {
        "category": "acoustic",
        "name": "Upper Right Lobe",
        "guide": (
            "Place the microphone 10-15 cm below the collar bone on the "
            "patient's RIGHT side, front of the chest. Ask them to take "
            "normal breaths, then cough once."
        ),
        "meta": "Acoustic zone: Upper Right Lobe",
        "icon": "🎙️",
    },
    "ull": {
        "category": "acoustic",
        "name": "Upper Left Lobe",
        "guide": (
            "Place the microphone 10-15 cm below the collar bone on the "
            "patient's LEFT side, front of the chest. Ask them to take "
            "normal breaths, then cough once."
        ),
        "meta": "Acoustic zone: Upper Left Lobe",
        "icon": "🎙️",
    },
    "lrl": {
        "category": "acoustic",
        "name": "Lower Right Lobe",
        "guide": (
            "Place the microphone on the RIGHT lower chest, just above the "
            "last rib. Ask the patient to breathe in deeply then cough."
        ),
        "meta": "Acoustic zone: Lower Right Lobe",
        "icon": "🎙️",
    },
    "lll": {
        "category": "acoustic",
        "name": "Lower Left Lobe",
        "guide": (
            "Place the microphone on the LEFT lower chest, just above the "
            "last rib. Ask the patient to breathe in deeply then cough."
        ),
        "meta": "Acoustic zone: Lower Left Lobe",
        "icon": "🎙️",
    },
    # --- Camera zones ------------------------------------------------------ #
    "conjunctiva": {
        "category": "camera",
        "name": "Palpebral Conjunctiva (Lower Eyelid)",
        "guide": (
            "Gently pull the patient's lower eyelid down while they look up, "
            "and frame the red/pink inner lining (conjunctiva) in the camera. "
            "Good even lighting; avoid direct flash glare."
        ),
        "meta": "Camera zone: Palpebral Conjunctiva (Anemia)",
        "icon": "👁️",
    },
    "nailbed": {
        "category": "camera",
        "name": "Fingernail Bed (Anemia)",
        "guide": (
            "Ask for the patient's hand, palm facing you. Frame ONE fingernail "
            "and its pink bed in the camera, close enough to fill most of the "
            "frame. Keep the hand steady and well lit."
        ),
        "meta": "Camera zone: Fingernail Bed (Anemia)",
        "icon": "🖐️",
    },
    "forehead": {
        "category": "camera",
        "name": "Forehead framing (rPPG Pulse)",
        "guide": (
            "Frame the patient's forehead only, filling 40-60% of the frame. "
            "Face a steady light source, remove covering hair, and hold still "
            "for the whole recording. Avoid bright windows behind the patient."
        ),
        "meta": "Camera zone: Forehead (rPPG Pulse)",
        "icon": "🧠",
    },
}

# --------------------------------------------------------------------------- #
# Clinical triage advice
#
# Base advice dictionaries used to assemble the diagnostic alert card. Each
# risk band carries a human readable level, a colour and advice that is
# translated / spoken in the patient's language.
# --------------------------------------------------------------------------- #
RISK_LEVELS = {
    "normal": {
        "label": "Normal",
        "colour": "#16a34a",
        "advice": (
            "No immediate concern detected in this screening. Continue routine "
            "care and re-screen as per local guidelines."
        ),
    },
    "low": {
        "label": "Low Risk",
        "colour": "#ca8a04",
        "advice": (
            "Mild deviation detected. Advise the patient to rest, stay "
            "hydrated, and follow up with a health worker for a repeat check."
        ),
    },
    "moderate": {
        "label": "Moderate Risk",
        "colour": "#ea580c",
        "advice": (
            "A clear deviation was found. Refer the patient to the nearest "
            "primary health centre for a confirmatory examination soon."
        ),
    },
    "high": {
        "label": "High Risk",
        "colour": "#dc2626",
        "advice": (
            "A significant finding was detected. Seek urgent medical care at "
            "the nearest health facility today."
        ),
    },
}

# Advice keyed per diagnostic module. The app picks the matching band and merges
# the module-specific guidance with the general band advice.
CLINICAL_ADVICE = {
    "rppg": {
        "title": "Pulse (rPPG) & Stress",
        "bands": {
            "normal": "Pulse is within the normal resting range. Patient is calm.",
            "low": "Pulse is slightly outside the ideal range. Encourage rest.",
            "moderate": "Elevated pulse detected. Check for fever, hydration and distress.",
            "high": "Pulse is significantly high or irregular. Refer urgently.",
        },
        "action": "If the patient feels well and has eaten, re-check after 10 minutes of rest.",
    },
    "pallor": {
        "title": "Anemia & Pallor Screening",
        "bands": {
            "normal": "Healthy colour seen in the screening area. No pallor detected.",
            "low": "Slight reduction in colour may be present. Offer iron-rich food advice.",
            "moderate": "Pallor likely present. Arrange a haemoglobin check at the facility.",
            "high": "Pallor strongly likely. Refer for early haemoglobin testing and review.",
        },
        "action": "Pallor is best confirmed by a lab haemoglobin test; share this screen with the clinician.",
    },
    "cough": {
        "title": "Cough Acoustic Profile",
        "bands": {
            "normal": "Cough acoustic features do not suggest a wet / infected pattern.",
            "low": "Mild acoustic changes seen. Watch for fever or breathing difficulty.",
            "moderate": "Wet-cough features present. Consider chest assessment and hydration.",
            "high": "Strong wet / productive pattern. Refer for respiratory evaluation soon.",
        },
        "action": "Monitor temperature and breathing; re-check if symptoms worsen.",
    },
    "tachypnea": {
        "title": "Tachypnea Counter (Breathing)",
        "bands": {
            "normal": "Breathing rate is within the expected range for this age.",
            "low": "Breathing rate is at the upper edge of normal.",
            "moderate": "Breathing rate is raised. Check for fever and restlessness.",
            "high": "Breathing rate is high (tachypnea). Refer for urgent assessment.",
        },
        "action": "Count breathing again after the child is calm; fever raises the rate.",
    },
}

# --------------------------------------------------------------------------- #
# Triage actions used in the exported CSV summary
# --------------------------------------------------------------------------- #
TRIAGE_ACTIONS = {
    "normal": "Routine care & re-screen",
    "low": "Follow-up advised",
    "moderate": "Refer to PHC soon",
    "high": "Urgent referral",
}

# Default export filename base (timestamp is appended by the exporter).
EXPORT_FILENAME_BASE = "nova_triage_report"

# Voice command words the hands-free assistant listens for (English).
# Their local-language equivalents are kept here for future extension.
VOICE_COMMANDS = {
    "record": ("record", "start"),
    "scan": ("scan", "capture"),
    "analyze": ("analyze", "analyse", "go"),
    "reset": ("reset", "clear"),
}
