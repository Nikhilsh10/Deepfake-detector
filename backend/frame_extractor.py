"""
Frame extractor for the Deepfake Detection System.

Uses OpenCV to pull frames from a video file at a configurable interval
and provides a helper to pre‑process a single image for model inference.
"""

import os
from typing import Optional

import cv2
import torch
from PIL import Image
from torchvision import transforms

from backend.config import DEVICE, FRAME_INTERVAL, MAX_FRAMES, PROCESSED_DIR


def extract_frames(
    video_path: str,
    output_dir: Optional[str] = None,
) -> list[str]:
    """Extract frames from a video at a fixed interval.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the source video.
    output_dir : str | None
        Directory where extracted JPEG frames will be saved.  Defaults to
        ``PROCESSED_DIR``.

    Returns
    -------
    list[str]
        Paths to the saved frame images.
    """
    if output_dir is None:
        output_dir = PROCESSED_DIR

    os.makedirs(output_dir, exist_ok=True)

    saved_paths: list[str] = []

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[frame_extractor] ERROR: Cannot open video {video_path}")
            return saved_paths

        frame_count: int = 0
        saved_count: int = 0

        while cap.isOpened() and saved_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_INTERVAL == 0:
                saved_count += 1
                frame_name = f"frame_{saved_count:04d}.jpg"
                frame_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                saved_paths.append(frame_path)

            frame_count += 1

        cap.release()

    except Exception as exc:
        print(f"[frame_extractor] ERROR processing video: {exc}")

    return saved_paths


def preprocess_image(
    image_path: str,
    transform: transforms.Compose,
) -> torch.Tensor:
    """Load an image, apply transforms, and return a batched tensor.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    transform : transforms.Compose
        Torchvision transform pipeline (from ``model_loader.get_transforms``).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(1, C, H, W)`` on ``DEVICE``.
    """
    image = Image.open(image_path).convert("RGB")
    tensor: torch.Tensor = transform(image).unsqueeze(0).to(DEVICE)
    return tensor
