"""
SQLite database module for the Deepfake Detection System.

Persists every analysis result so they can be queried later via the
``/stats`` and ``/history`` API endpoints.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

from backend.config import BASE_DIR

DB_PATH: str = os.path.join(BASE_DIR, "data", "deepfake_results.db")


def _get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.Connection(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the results table if it does not exist."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                filename      TEXT    NOT NULL,
                file_type     TEXT    NOT NULL,       -- 'image' | 'video'
                prediction    TEXT    NOT NULL,
                confidence    REAL    NOT NULL,
                fake_prob     REAL    NOT NULL,
                real_prob     REAL    NOT NULL,
                total_frames  INTEGER DEFAULT 0,
                fake_frames   INTEGER DEFAULT 0,
                real_frames   INTEGER DEFAULT 0,
                processing_s  REAL    DEFAULT 0.0,
                details_json  TEXT,                   -- full result dict as JSON
                created_at    TEXT    NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def save_result(
    filename: str,
    file_type: str,
    result: dict[str, Any],
) -> int:
    """Insert an analysis result and return the new row id.

    Parameters
    ----------
    filename : str
        Original uploaded file name.
    file_type : str
        ``'image'`` or ``'video'``.
    result : dict
        The full result dictionary returned by the detector.

    Returns
    -------
    int
        Auto‑incremented row id.
    """
    conn = _get_connection()
    try:
        cur = conn.execute(
            """
            INSERT INTO results
                (filename, file_type, prediction, confidence,
                 fake_prob, real_prob, total_frames, fake_frames,
                 real_frames, processing_s, details_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                file_type,
                result.get("prediction", "UNKNOWN"),
                result.get("confidence", 0.0),
                result.get("fake_probability", 0.0),
                result.get("real_probability", 0.0),
                result.get("total_frames_analyzed", 0),
                result.get("fake_frames", 0),
                result.get("real_frames", 0),
                result.get("processing_time_seconds", 0.0),
                json.dumps(result, default=str),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid or 0
    finally:
        conn.close()


def get_stats() -> dict[str, Any]:
    """Return aggregate statistics from the database.

    Returns
    -------
    dict
        Keys: ``total_images_processed``, ``total_videos_processed``,
        ``total_fakes_detected``, ``total_reals_detected``,
        ``average_confidence``.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN file_type = 'image' THEN 1 ELSE 0 END), 0) AS images,
                COALESCE(SUM(CASE WHEN file_type = 'video' THEN 1 ELSE 0 END), 0) AS videos,
                COALESCE(SUM(CASE WHEN prediction = 'FAKE'  THEN 1 ELSE 0 END), 0) AS fakes,
                COALESCE(SUM(CASE WHEN prediction = 'REAL'  THEN 1 ELSE 0 END), 0) AS reals,
                COALESCE(AVG(confidence), 0.0) AS avg_conf
            FROM results
            """
        ).fetchone()

        return {
            "total_images_processed": row["images"],
            "total_videos_processed": row["videos"],
            "total_fakes_detected": row["fakes"],
            "total_reals_detected": row["reals"],
            "average_confidence": round(row["avg_conf"], 4),
        }
    finally:
        conn.close()


def get_history(
    limit: int = 50,
    offset: int = 0,
    file_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return recent analysis results, newest first.

    Parameters
    ----------
    limit : int
        Maximum number of records to return.
    offset : int
        Number of records to skip (for pagination).
    file_type : str | None
        Filter by ``'image'`` or ``'video'``.  ``None`` returns all.

    Returns
    -------
    list[dict]
    """
    conn = _get_connection()
    try:
        query = "SELECT * FROM results"
        params: list[Any] = []

        if file_type:
            query += " WHERE file_type = ?"
            params.append(file_type)

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
