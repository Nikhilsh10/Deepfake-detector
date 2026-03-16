"""
Configuration module for the Deepfake Detection System.

All constants, paths, model settings, and allowed file types are
centralised here so that every other module imports from one place.
"""

import os
import torch

# ── Directory paths ──────────────────────────────────────────────────────────
BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR: str = os.path.join(BASE_DIR, "data", "uploads")
PROCESSED_DIR: str = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR: str = os.path.join(BASE_DIR, "models")

# ── Model configuration ─────────────────────────────────────────────────────
MODEL_NAME: str = "efficientnet_b4"
PRETRAINED: bool = True
NUM_CLASSES: int = 2
IMAGE_SIZE: int = 224
BATCH_SIZE: int = 8
CONFIDENCE_THRESHOLD: float = 0.5

# ── Frame extraction ────────────────────────────────────────────────────────
MAX_FRAMES: int = 20
FRAME_INTERVAL: int = 10

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── Labels ───────────────────────────────────────────────────────────────────
FAKE_LABEL: int = 1
REAL_LABEL: int = 0
CLASS_NAMES: list[str] = ["REAL", "FAKE"]

# ── API settings ─────────────────────────────────────────────────────────────
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000

# ── Allowed file types ──────────────────────────────────────────────────────
ALLOWED_VIDEO_TYPES: list[str] = [".mp4", ".avi", ".mov", ".mkv"]
ALLOWED_IMAGE_TYPES: list[str] = [".jpg", ".jpeg", ".png", ".webp"]
