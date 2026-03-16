"""
Core detection logic for the Deepfake Detection System.

Orchestrates model inference on single images and full videos,
aggregating per‑frame predictions into an overall verdict.
"""

import os
import time
import uuid

import torch
import torch.nn.functional as F

from backend.config import (
    CLASS_NAMES,
    CONFIDENCE_THRESHOLD,
    FAKE_LABEL,
    PROCESSED_DIR,
    REAL_LABEL,
)
from backend.face_detector import detect_and_crop_face
from backend.frame_extractor import extract_frames, preprocess_image
from backend.heatmap import generate_heatmap
from backend.model_loader import get_transforms, load_model


def analyze_image(image_path: str) -> dict:
    """Run deepfake detection on a single image.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    dict
        Detection result including prediction, confidence,
        class probabilities, and heatmap path.
    """
    model = load_model()
    transform = get_transforms()

    # Face crop (use the cropped image for inference, original for heatmap) ─
    cropped_dir = os.path.join(PROCESSED_DIR, "faces")
    os.makedirs(cropped_dir, exist_ok=True)
    crop_name = f"face_{uuid.uuid4().hex[:8]}.jpg"
    crop_path = os.path.join(cropped_dir, crop_name)
    analysis_path: str = detect_and_crop_face(image_path, output_path=crop_path)

    # Pre‑process ────────────────────────────────────────────────────────────
    image_tensor: torch.Tensor = preprocess_image(analysis_path, transform)

    # Inference ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        outputs: torch.Tensor = model(image_tensor)
        probabilities: torch.Tensor = F.softmax(outputs, dim=1)

    fake_prob: float = probabilities[0][FAKE_LABEL].item()
    real_prob: float = probabilities[0][REAL_LABEL].item()
    predicted_class: int = torch.argmax(probabilities, dim=1).item()
    confidence: float = probabilities[0][predicted_class].item()
    prediction: str = CLASS_NAMES[predicted_class]

    # Heatmap ────────────────────────────────────────────────────────────────
    heatmap_dir = os.path.join(PROCESSED_DIR, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    heatmap_filename = f"heatmap_{uuid.uuid4().hex[:8]}.jpg"
    heatmap_output = os.path.join(heatmap_dir, heatmap_filename)

    heatmap_path: str = generate_heatmap(
        model, image_tensor, image_path, heatmap_output
    )

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
        "heatmap_path": heatmap_path,
        "frame_path": image_path,
    }


def analyze_video(video_path: str) -> dict:
    """Run deepfake detection on a video.

    Extracts frames from the video, analyses each one individually,
    and aggregates the results into an overall verdict.

    Parameters
    ----------
    video_path : str
        Path to the video file.

    Returns
    -------
    dict
        Aggregated detection result with per‑frame breakdown.
    """
    start_time: float = time.time()

    # Create a unique output directory for this video's frames ──────────────
    video_id: str = uuid.uuid4().hex[:8]
    output_dir: str = os.path.join(PROCESSED_DIR, f"video_{video_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames ────────────────────────────────────────────────────────
    frame_paths: list[str] = extract_frames(video_path, output_dir)

    if not frame_paths:
        return {
            "prediction": "UNKNOWN",
            "confidence": 0.0,
            "fake_probability": 0.0,
            "real_probability": 0.0,
            "total_frames_analyzed": 0,
            "fake_frames": 0,
            "real_frames": 0,
            "frame_results": [],
            "processing_time_seconds": round(time.time() - start_time, 2),
        }

    # Analyse each frame ────────────────────────────────────────────────────
    frame_results: list[dict] = []
    fake_count: int = 0
    real_count: int = 0
    total_fake_prob: float = 0.0
    total_confidence: float = 0.0

    for frame_path in frame_paths:
        result: dict = analyze_image(frame_path)
        frame_results.append(result)

        if result["prediction"] == CLASS_NAMES[FAKE_LABEL]:
            fake_count += 1
        else:
            real_count += 1

        total_fake_prob += result["fake_probability"]
        total_confidence += result["confidence"]

    # Aggregate ──────────────────────────────────────────────────────────────
    num_frames: int = len(frame_results)
    avg_fake_prob: float = total_fake_prob / num_frames
    avg_confidence: float = total_confidence / num_frames
    overall_prediction: str = (
        CLASS_NAMES[FAKE_LABEL]
        if avg_fake_prob > CONFIDENCE_THRESHOLD
        else CLASS_NAMES[REAL_LABEL]
    )

    processing_time: float = round(time.time() - start_time, 2)

    return {
        "prediction": overall_prediction,
        "confidence": round(avg_confidence, 4),
        "fake_probability": round(avg_fake_prob, 4),
        "real_probability": round(1.0 - avg_fake_prob, 4),
        "total_frames_analyzed": num_frames,
        "fake_frames": fake_count,
        "real_frames": real_count,
        "frame_results": frame_results,
        "processing_time_seconds": processing_time,
    }
