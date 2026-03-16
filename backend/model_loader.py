"""
Model loader for the Deepfake Detection System.

Provides helpers to build an EfficientNet‑B4 classifier via *timm* and
to create the image‑transform pipeline used during inference.
"""

import os
from typing import Optional

import timm
import torch
import torch.nn as nn
from torchvision import transforms

from backend.config import (
    DEVICE,
    IMAGE_SIZE,
    MODEL_DIR,
    MODEL_NAME,
    NUM_CLASSES,
    PRETRAINED,
)

# Module‑level cache so the model is loaded only once.
_cached_model: Optional[nn.Module] = None
_cached_transforms: Optional[transforms.Compose] = None


def load_model(model_path: Optional[str] = None) -> nn.Module:
    """Load an EfficientNet‑B4 model for binary classification.

    Parameters
    ----------
    model_path : str | None
        Path to a ``.pth`` checkpoint.  When *None* the function looks for
        ``<MODEL_DIR>/deepfake_efficientnet_b4.pth``.  If that file does
        not exist, ImageNet‑pretrained weights are used as a demo fallback.

    Returns
    -------
    nn.Module
        The model in **eval** mode, moved to ``DEVICE``.
    """
    global _cached_model  # noqa: PLW0603

    if _cached_model is not None:
        return _cached_model

    # Build backbone ─────────────────────────────────────────────────────────
    model: nn.Module = timm.create_model(
        MODEL_NAME,
        pretrained=PRETRAINED,
        num_classes=0,  # remove default head
    )

    # Determine in‑features of the original classifier ──────────────────────
    in_features: int = model.num_features

    # Replace classifier head ───────────────────────────────────────────────
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, NUM_CLASSES),
    )

    # Optionally load fine‑tuned weights ────────────────────────────────────
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, f"deepfake_{MODEL_NAME}.pth")

    if os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"[model_loader] Loaded weights from {model_path}")
    else:
        print(
            "[model_loader] No saved weights found – using ImageNet "
            "pretrained weights (demo mode)."
        )

    model.eval()
    model.to(DEVICE)

    _cached_model = model
    return model


def get_transforms() -> transforms.Compose:
    """Return the image‑transform pipeline for inference.

    The pipeline resizes, converts to tensor, and normalises with ImageNet
    statistics.

    Returns
    -------
    transforms.Compose
    """
    global _cached_transforms  # noqa: PLW0603

    if _cached_transforms is not None:
        return _cached_transforms

    _cached_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return _cached_transforms
