import sqlite3
import json
import os
from datetime import datetime
from config import DB_PATH, DATA_DIR


def get_connection():
    """Get a connection to the SQLite database."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            source_url TEXT,
            device_index INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            confidence REAL,
            snapshot_path TEXT,
            description TEXT,
            objects_detected TEXT,
            FOREIGN KEY (camera_id) REFERENCES cameras(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_name TEXT,
            status TEXT DEFAULT 'queued',
            total_frames INTEGER,
            processed_frames INTEGER DEFAULT 0,
            incidents_found INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            upload_id INTEGER,
            frame_number INTEGER,
            timestamp_in_video REAL,
            confidence REAL,
            snapshot_path TEXT,
            objects_detected TEXT,
            FOREIGN KEY (upload_id) REFERENCES video_uploads(id)
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# Camera CRUD
# ============================================================

def insert_camera(name, source_type, source_url=None, device_index=0):
    """Add a new camera source."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO cameras (name, source_type, source_url, device_index) VALUES (?, ?, ?, ?)",
        (name, source_type, source_url, device_index)
    )
    conn.commit()
    camera_id = cursor.lastrowid
    conn.close()
    return camera_id


def get_cameras():
    """Get all registered cameras."""
    conn = get_connection()
    cameras = conn.execute("SELECT * FROM cameras ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(c) for c in cameras]


def get_camera(camera_id):
    """Get a single camera by ID."""
    conn = get_connection()
    camera = conn.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)).fetchone()
    conn.close()
    return dict(camera) if camera else None


def delete_camera(camera_id):
    """Delete a camera source."""
    conn = get_connection()
    conn.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
    conn.commit()
    conn.close()


# ============================================================
# Incident CRUD
# ============================================================

def insert_incident(camera_id, confidence, snapshot_path, description, objects_detected):
    """Log a new detected incident."""
    conn = get_connection()
    cursor = conn.cursor()
    objects_json = json.dumps(objects_detected) if isinstance(objects_detected, list) else objects_detected
    cursor.execute(
        "INSERT INTO incidents (camera_id, confidence, snapshot_path, description, objects_detected) "
        "VALUES (?, ?, ?, ?, ?)",
        (camera_id, confidence, snapshot_path, description, objects_json)
    )
    conn.commit()
    incident_id = cursor.lastrowid
    conn.close()
    return incident_id


def get_incidents(limit=50, camera_id=None, date_from=None, date_to=None):
    """Get incidents with optional filters."""
    conn = get_connection()
    query = "SELECT * FROM incidents WHERE 1=1"
    params = []

    if camera_id is not None:
        query += " AND camera_id = ?"
        params.append(camera_id)
    if date_from is not None:
        query += " AND timestamp >= ?"
        params.append(date_from)
    if date_to is not None:
        query += " AND timestamp <= ?"
        params.append(date_to)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    incidents = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(i) for i in incidents]


def get_incident(incident_id):
    """Get a single incident by ID."""
    conn = get_connection()
    incident = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    conn.close()
    return dict(incident) if incident else None


def get_incident_count(today_only=False):
    """Get total incident count, optionally for today only."""
    conn = get_connection()
    if today_only:
        today = datetime.now().strftime("%Y-%m-%d")
        result = conn.execute(
            "SELECT COUNT(*) as count FROM incidents WHERE date(timestamp) = ?", (today,)
        ).fetchone()
    else:
        result = conn.execute("SELECT COUNT(*) as count FROM incidents").fetchone()
    conn.close()
    return result["count"]


# ============================================================
# Video Upload CRUD
# ============================================================

def insert_video_upload(filename, original_name, total_frames=0):
    """Create a new video upload record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO video_uploads (filename, original_name, total_frames) VALUES (?, ?, ?)",
        (filename, original_name, total_frames)
    )
    conn.commit()
    upload_id = cursor.lastrowid
    conn.close()
    return upload_id


def update_video_upload(upload_id, status=None, processed_frames=None, incidents_found=None):
    """Update a video upload's processing status."""
    conn = get_connection()
    updates = []
    params = []

    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if processed_frames is not None:
        updates.append("processed_frames = ?")
        params.append(processed_frames)
    if incidents_found is not None:
        updates.append("incidents_found = ?")
        params.append(incidents_found)

    if updates:
        params.append(upload_id)
        conn.execute(
            f"UPDATE video_uploads SET {', '.join(updates)} WHERE id = ?",
            params
        )
        conn.commit()
    conn.close()


def get_video_uploads():
    """Get all video uploads."""
    conn = get_connection()
    uploads = conn.execute("SELECT * FROM video_uploads ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(u) for u in uploads]


def get_video_upload(upload_id):
    """Get a single video upload by ID."""
    conn = get_connection()
    upload = conn.execute("SELECT * FROM video_uploads WHERE id = ?", (upload_id,)).fetchone()
    conn.close()
    return dict(upload) if upload else None


# ============================================================
# Video Detection CRUD
# ============================================================

def insert_video_detection(upload_id, frame_number, timestamp_in_video, confidence,
                           snapshot_path, objects_detected):
    """Log a detection found in an uploaded video."""
    conn = get_connection()
    cursor = conn.cursor()
    objects_json = json.dumps(objects_detected) if isinstance(objects_detected, list) else objects_detected
    cursor.execute(
        "INSERT INTO video_detections "
        "(upload_id, frame_number, timestamp_in_video, confidence, snapshot_path, objects_detected) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (upload_id, frame_number, timestamp_in_video, confidence, snapshot_path, objects_json)
    )
    conn.commit()
    detection_id = cursor.lastrowid
    conn.close()
    return detection_id


def get_video_detections(upload_id):
    """Get all detections for a given video upload."""
    conn = get_connection()
    detections = conn.execute(
        "SELECT * FROM video_detections WHERE upload_id = ? ORDER BY frame_number",
        (upload_id,)
    ).fetchall()
    conn.close()
    return [dict(d) for d in detections]


# Initialize the database on import
init_db()
