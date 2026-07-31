"""
src/storage.py
--------------
SQLite persistence layer for processed traffic records.

The database file lives at data/traffic_log.db (intentionally NOT in
.gitignore so it persists across deployments for demo purposes).

Table schema: traffic_records
  id              INTEGER PRIMARY KEY AUTOINCREMENT
  timestamp       TEXT    — ISO-8601 timestamp of when the record was processed
  ground_truth    TEXT    — original NSL-KDD label (normal / attack type)
  is_anomaly      INTEGER — 1 if model flagged as anomaly, 0 otherwise
  anomaly_score   REAL    — negated score_samples (higher = more suspicious)
  prediction_raw  INTEGER — raw model output (1 = normal, -1 = anomaly)
  protocol_type   TEXT
  service         TEXT
  flag            TEXT
  src_bytes       REAL
  dst_bytes       REAL
  duration        REAL
  count           REAL
  srv_count       REAL
  serror_rate     REAL
  rerror_rate     REAL
"""

import os
import sqlite3
import pandas as pd
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# Path to the database
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH   = os.path.join(_BASE_DIR, "data", "traffic_log.db")

# Subset of raw features stored alongside predictions (for auditability)
_STORED_FEATURES = [
    "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "duration",
    "count", "srv_count", "serror_rate", "rerror_rate",
]


@contextmanager
def _get_conn():
    """Context manager that yields an open SQLite connection and auto-commits."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the database file and the traffic_records table if they don't
    exist yet. Safe to call multiple times (idempotent)."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS traffic_records (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT    NOT NULL,
                ground_truth   TEXT    NOT NULL,
                is_anomaly     INTEGER NOT NULL,
                anomaly_score  REAL    NOT NULL,
                prediction_raw INTEGER NOT NULL,
                protocol_type  TEXT,
                service        TEXT,
                flag           TEXT,
                src_bytes      REAL,
                dst_bytes      REAL,
                duration       REAL,
                count          REAL,
                srv_count      REAL,
                serror_rate    REAL,
                rerror_rate    REAL
            )
        """)


def insert_record(
    timestamp: str,
    ground_truth: str,
    prediction: dict,
    raw_row: dict,
) -> None:
    """Append a processed record to traffic_records.

    Args:
        timestamp:    ISO-8601 string of when the record was processed.
        ground_truth: Original NSL-KDD label (e.g. "normal", "neptune").
        prediction:   Output of preprocessing.predict_record().
        raw_row:      Original dict with the 43 NSL-KDD fields.
    """
    row = (
        timestamp,
        ground_truth,
        int(prediction["is_anomaly"]),
        float(prediction["anomaly_score"]),
        int(prediction["prediction_raw"]),
        raw_row.get("protocol_type"),
        raw_row.get("service"),
        raw_row.get("flag"),
        raw_row.get("src_bytes"),
        raw_row.get("dst_bytes"),
        raw_row.get("duration"),
        raw_row.get("count"),
        raw_row.get("srv_count"),
        raw_row.get("serror_rate"),
        raw_row.get("rerror_rate"),
    )
    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO traffic_records (
                timestamp, ground_truth, is_anomaly, anomaly_score,
                prediction_raw, protocol_type, service, flag,
                src_bytes, dst_bytes, duration, count, srv_count,
                serror_rate, rerror_rate
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, row)


def get_last_n(n: int = 20) -> pd.DataFrame:
    """Retrieve the N most recently inserted records as a DataFrame.

    Returns an empty DataFrame with the correct columns if the table is empty.
    """
    with _get_conn() as conn:
        df = pd.read_sql_query(
            f"SELECT * FROM traffic_records ORDER BY id DESC LIMIT {n}",
            conn,
        )
    return df


def get_all_for_retraining() -> pd.DataFrame:
    """Return ALL stored records — intended for use in future retraining
    pipelines. Includes ground truth label and all stored features."""
    with _get_conn() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM traffic_records ORDER BY id ASC",
            conn,
        )
    return df


def get_total_count() -> int:
    """Return the total number of records in traffic_records (across all
    sessions, not just the current one)."""
    with _get_conn() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM traffic_records")
        return cur.fetchone()[0]
