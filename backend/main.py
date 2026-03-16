"""
FastAPI application for the Deepfake Detection System.

Provides REST endpoints for image, video, webcam, and batch deepfake
detection, plus helpers to serve heatmaps, frames, reports, history,
health, and stats.  Includes optional API‑key authentication and
SQLite result logging.
"""

import base64
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.auth import APIKeyMiddleware
from backend.config import (
    ALLOWED_IMAGE_TYPES,
    ALLOWED_VIDEO_TYPES,
    API_HOST,
    API_PORT,
    DEVICE,
    MODEL_NAME,
    PROCESSED_DIR,
    UPLOAD_DIR,
)
from backend.database import get_history, get_stats as db_get_stats, init_db, save_result
from backend.detector import analyze_image, analyze_video
from backend.report import generate_pdf_report


# ── Lifecycle ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create required directories and initialise the database on startup."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "reports"), exist_ok=True)
    init_db()
    print("[main] Directories created / verified.  Database initialised.")
    yield


app = FastAPI(
    title="Deepfake Detection API",
    description="Real‑time deepfake detection using EfficientNet‑B4 and Grad‑CAM.",
    version="2.0.0",
    lifespan=lifespan,
)

# ── Middleware stack ────────────────────────────────────────────────────────
# 1. CORS (outermost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. API key auth (optional — disabled when API_KEY env var is unset)
app.add_middleware(APIKeyMiddleware)


# ── Request timing middleware ──────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Log and attach an ``X-Process-Time`` header to every response."""
    start = time.time()
    response = await call_next(request)
    process_time = time.time() - start
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    print(f"[timing] {request.method} {request.url.path} — {process_time:.4f}s")
    return response


# ── Pydantic models ────────────────────────────────────────────────────────
class WebcamPayload(BaseModel):
    """Body schema for the webcam detection endpoint."""

    image_base64: str


# ── Helpers ─────────────────────────────────────────────────────────────────
def _validate_extension(filename: str, allowed: list[str]) -> str:
    """Return the lower‑cased file extension or raise 400."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Accepted: {allowed}",
        )
    return ext


async def _save_upload(upload: UploadFile, dest_dir: str) -> str:
    """Persist an uploaded file and return its path on disk."""
    os.makedirs(dest_dir, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex[:8]}_{upload.filename}"
    file_path = os.path.join(dest_dir, unique_name)
    contents = await upload.read()
    with open(file_path, "wb") as f:
        f.write(contents)
    return file_path


# ── Endpoints ───────────────────────────────────────────────────────────────


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)) -> JSONResponse:
    """Detect deepfake in an uploaded image."""
    try:
        _validate_extension(file.filename or "", ALLOWED_IMAGE_TYPES)
        file_path: str = await _save_upload(file, UPLOAD_DIR)
        result: dict = analyze_image(file_path)
        save_result(os.path.basename(file_path), "image", result)
        return JSONResponse(content={**result, "filename": os.path.basename(file_path)})

    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "Image analysis failed", "detail": str(exc)},
        )


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)) -> JSONResponse:
    """Detect deepfake in an uploaded video."""
    try:
        _validate_extension(file.filename or "", ALLOWED_VIDEO_TYPES)
        file_path: str = await _save_upload(file, UPLOAD_DIR)
        result: dict = analyze_video(file_path)
        save_result(os.path.basename(file_path), "video", result)
        return JSONResponse(content={**result, "filename": os.path.basename(file_path)})

    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "Video analysis failed", "detail": str(exc)},
        )


@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...)) -> JSONResponse:
    """Batch detection: analyse multiple images/videos in one request.

    Returns a list of individual results.
    """
    try:
        results: list[dict] = []
        for upload in files:
            fname = upload.filename or ""
            ext = os.path.splitext(fname)[1].lower()
            file_path = await _save_upload(upload, UPLOAD_DIR)

            if ext in ALLOWED_VIDEO_TYPES:
                result = analyze_video(file_path)
                save_result(os.path.basename(file_path), "video", result)
            elif ext in ALLOWED_IMAGE_TYPES:
                result = analyze_image(file_path)
                save_result(os.path.basename(file_path), "image", result)
            else:
                results.append({"filename": fname, "error": f"Unsupported type: {ext}"})
                continue

            results.append({**result, "filename": os.path.basename(file_path)})

        return JSONResponse(content={"results": results, "total": len(results)})

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "Batch analysis failed", "detail": str(exc)},
        )


@app.get("/heatmap/{filename}")
async def get_heatmap(filename: str) -> FileResponse:
    """Serve a Grad‑CAM heatmap image by filename."""
    heatmap_path = os.path.join(PROCESSED_DIR, "heatmaps", filename)
    if not os.path.isfile(heatmap_path):
        raise HTTPException(status_code=404, detail="Heatmap file not found.")
    return FileResponse(heatmap_path, media_type="image/jpeg")


@app.get("/frame/{filename:path}")
async def get_frame(filename: str) -> FileResponse:
    """Serve an extracted frame image by filename (supports nested paths)."""
    frame_path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.isfile(frame_path):
        raise HTTPException(status_code=404, detail="Frame file not found.")
    return FileResponse(frame_path, media_type="image/jpeg")


@app.post("/detect/webcam")
async def detect_webcam(payload: WebcamPayload) -> JSONResponse:
    """Detect deepfake from a base64‑encoded webcam frame."""
    temp_path: str = ""
    try:
        image_data: bytes = base64.b64decode(payload.image_base64)
        temp_path = os.path.join(UPLOAD_DIR, f"webcam_{uuid.uuid4().hex[:8]}.jpg")
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        with open(temp_path, "wb") as f:
            f.write(image_data)

        result: dict = analyze_image(temp_path)
        save_result("webcam_frame", "image", result)
        return JSONResponse(content=result)

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "Webcam analysis failed", "detail": str(exc)},
        )
    finally:
        if temp_path and os.path.isfile(temp_path):
            os.remove(temp_path)


@app.post("/report")
async def create_report(file: UploadFile = File(...)) -> FileResponse:
    """Analyse an image and return a downloadable visual report."""
    try:
        _validate_extension(file.filename or "", ALLOWED_IMAGE_TYPES)
        file_path = await _save_upload(file, UPLOAD_DIR)
        result = analyze_image(file_path)
        save_result(os.path.basename(file_path), "image", result)

        report_name = f"report_{uuid.uuid4().hex[:8]}.png"
        report_path = os.path.join(PROCESSED_DIR, "reports", report_name)

        generate_pdf_report(
            result=result,
            filename=os.path.basename(file_path),
            output_path=report_path,
            heatmap_path=result.get("heatmap_path"),
        )

        return FileResponse(
            report_path,
            media_type="image/png",
            filename=report_name,
        )

    except HTTPException:
        raise
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "Report generation failed", "detail": str(exc)},
        )


@app.get("/health")
async def health_check() -> JSONResponse:
    """Return service health status."""
    return JSONResponse(
        content={
            "status": "ok",
            "device": DEVICE,
            "model": MODEL_NAME,
        }
    )


@app.get("/stats")
async def get_stats_endpoint() -> JSONResponse:
    """Return processing statistics from the database."""
    return JSONResponse(content=db_get_stats())


@app.get("/history")
async def get_history_endpoint(
    limit: int = 50,
    offset: int = 0,
    file_type: str | None = None,
) -> JSONResponse:
    """Return recent analysis results with optional pagination and filtering."""
    return JSONResponse(
        content={
            "results": get_history(limit=limit, offset=offset, file_type=file_type),
            "limit": limit,
            "offset": offset,
        }
    )


# ── Dev entry‑point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=True)
