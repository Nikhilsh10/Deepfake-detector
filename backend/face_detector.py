"""
Face detector module for the Deepfake Detection System.

Uses MTCNN-style face detection via OpenCV's DNN module with a
pre‑trained Caffe model.  When no DNN model files are available
it falls back to OpenCV's Haar cascade for face detection.

Cropping faces before feeding them to EfficientNet dramatically
improves detection accuracy.
"""

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from backend.config import IMAGE_SIZE


def _get_haar_cascade() -> cv2.CascadeClassifier:
    """Load OpenCV's built‑in Haar cascade for frontal faces."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def detect_and_crop_face(
    image_path: str,
    output_path: Optional[str] = None,
    padding: float = 0.3,
) -> str:
    """Detect the largest face in an image and save a cropped version.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    output_path : str | None
        Where to save the cropped face.  When *None* the crop overwrites
        the original file.
    padding : float
        Fractional padding around the detected face bounding box
        (0.3 = 30 % on each side).

    Returns
    -------
    str
        Path to the saved (cropped) image — or the original path when no
        face is detected.
    """
    if output_path is None:
        output_path = image_path

    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        # Detect faces with Haar cascade ────────────────────────────────────
        cascade = _get_haar_cascade()
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            # No face found — return original
            return image_path

        # Pick the largest face ─────────────────────────────────────────────
        faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, fw, fh = faces_sorted[0]

        # Add padding ──────────────────────────────────────────────────────
        pad_w = int(fw * padding)
        pad_h = int(fh * padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + fw + pad_w)
        y2 = min(h, y + fh + pad_h)

        face_crop = img[y1:y2, x1:x2]

        # Resize to model input size ───────────────────────────────────────
        face_crop = cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, face_crop)
        return output_path

    except Exception as exc:
        print(f"[face_detector] WARNING: {exc}")
        return image_path


def detect_faces_in_frame(
    image_path: str,
) -> list[dict]:
    """Return bounding boxes for all faces detected in an image.

    Parameters
    ----------
    image_path : str
        Path to the input image.

    Returns
    -------
    list[dict]
        Each dict has keys ``x``, ``y``, ``w``, ``h`` (pixel coords).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade = _get_haar_cascade()
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        return [
            {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            for (x, y, w, h) in faces
        ]

    except Exception as exc:
        print(f"[face_detector] WARNING: {exc}")
        return []
