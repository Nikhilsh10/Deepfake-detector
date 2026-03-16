"""
Grad‑CAM heatmap generator for the Deepfake Detection System.

Overlays a class‑activation heatmap on the original image so the user
can see *which regions* the model focused on when making its decision.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def generate_heatmap(
    model: nn.Module,
    image_tensor: torch.Tensor,
    image_path: str,
    output_path: str,
) -> str:
    """Generate a Grad‑CAM heatmap and overlay it on the source image.

    Parameters
    ----------
    model : nn.Module
        The EfficientNet model (in eval mode).
    image_tensor : torch.Tensor
        Pre‑processed tensor of shape ``(1, C, H, W)``.
    image_path : str
        Path to the original (un‑transformed) image – used as the
        background for the overlay.
    output_path : str
        Where to save the resulting heatmap overlay image.

    Returns
    -------
    str
        ``output_path`` on success, or ``image_path`` if heatmap
        generation fails for any reason.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Determine the target layer (last convolutional block) ─────────────
        # timm EfficientNet models expose `.blocks` as a Sequential of blocks.
        if hasattr(model, "blocks"):
            target_layers = [model.blocks[-1]]
        elif hasattr(model, "features"):
            target_layers = [model.features[-1]]
        else:
            # Fallback – return original image
            print("[heatmap] WARNING: Could not determine target layer.")
            return image_path

        # Build Grad‑CAM ────────────────────────────────────────────────────
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam: np.ndarray = cam(
            input_tensor=image_tensor,
            targets=None,  # uses the model's predicted class
        )
        grayscale_cam = grayscale_cam[0, :]  # first image in batch

        # Load and resize original image to match cam dimensions ────────────
        original_image = Image.open(image_path).convert("RGB")
        original_image = original_image.resize(
            (grayscale_cam.shape[1], grayscale_cam.shape[0])
        )
        rgb_image = np.array(original_image, dtype=np.float32) / 255.0

        # Overlay heatmap on the original image ─────────────────────────────
        overlay: np.ndarray = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET,
        )

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, overlay_bgr)

        return output_path

    except Exception as exc:
        print(f"[heatmap] ERROR generating heatmap: {exc}")
        return image_path
