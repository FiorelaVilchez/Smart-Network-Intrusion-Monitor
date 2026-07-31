"""
src/simulator.py
----------------
Sequential CSV reader with a persistent file-pointer stored in
st.session_state. Reads KDDTest+.txt one row at a time without loading the
entire file into memory (important: the NSL-KDD test file has 22,544 rows).

Design decisions:
  - We keep an open file handle in session_state to avoid re-opening on every
    Streamlit rerun. The handle is recreated only on Reset or first start.
  - If EOF is reached, the file loops back to the beginning automatically and
    a flag is set so the UI can show a "looped" notice.
  - Each yielded record is a plain dict {column_name: value} with proper types
    (numeric columns coerced to float/int).
"""

import os
import csv
import streamlit as st

from src.preprocessing import COLUMN_NAMES  # 43 column names in order

# ---------------------------------------------------------------------------
# Path to the test dataset
# ---------------------------------------------------------------------------
_BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_PATH = os.path.join(_BASE_DIR, "data", "KDDTest+.txt")

# Columns that should be stored as integers (boolean / count features)
_INT_COLS = {
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "dst_host_count", "dst_host_srv_count",
    "difficulty_level", "src_bytes", "dst_bytes", "duration",
}


def _coerce_row(raw: list) -> dict:
    """Convert a list of string values into a typed dict using COLUMN_NAMES."""
    record = {}
    for col, val in zip(COLUMN_NAMES, raw):
        val = val.strip()
        if col in _INT_COLS:
            record[col] = int(val)
        elif col in {"protocol_type", "service", "flag", "label"}:
            record[col] = val  # keep as string
        else:
            try:
                record[col] = float(val)
            except ValueError:
                record[col] = val
    return record


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
def init_simulator():
    """Initialise all simulator-related session_state keys if not present.
    Call once at app startup."""
    if "sim_file_handle" not in st.session_state:
        st.session_state.sim_file_handle   = None
    if "sim_running" not in st.session_state:
        st.session_state.sim_running       = False
    if "sim_row_count" not in st.session_state:
        st.session_state.sim_row_count     = 0   # total rows processed this session
    if "sim_alert_count" not in st.session_state:
        st.session_state.sim_alert_count   = 0
    if "sim_looped" not in st.session_state:
        st.session_state.sim_looped        = False  # True when file wrapped around
    if "sim_loop_count" not in st.session_state:
        st.session_state.sim_loop_count    = 0   # how many times file has looped
    if "sim_start_time" not in st.session_state:
        st.session_state.sim_start_time    = None
    if "sim_history" not in st.session_state:
        # In-memory ring buffer of the last 500 processed records for charts
        st.session_state.sim_history       = []


def reset_simulator():
    """Close the file handle and reset all counters. Called by the Reset button."""
    if st.session_state.get("sim_file_handle") is not None:
        try:
            st.session_state.sim_file_handle.close()
        except Exception:
            pass
    st.session_state.sim_file_handle   = None
    st.session_state.sim_running       = False
    st.session_state.sim_row_count     = 0
    st.session_state.sim_alert_count   = 0
    st.session_state.sim_looped        = False
    st.session_state.sim_loop_count    = 0
    st.session_state.sim_start_time    = None
    st.session_state.sim_history       = []


def _open_file() -> None:
    """Open (or reopen) the test dataset file and store the handle."""
    if not os.path.exists(_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {_DATA_PATH}. "
            "Please place KDDTest+.txt in the data/ directory."
        )
    fh = open(_DATA_PATH, "r", newline="", encoding="utf-8")
    st.session_state.sim_file_handle = fh


def read_next_record() -> dict | None:
    """Read the next raw record from the CSV file.

    Returns a dict {column_name: value} or None if the file is empty.
    Automatically loops to the beginning of the file on EOF.
    Sets st.session_state.sim_looped = True and increments sim_loop_count
    whenever a loop occurs.
    """
    # Open file lazily on first call
    if st.session_state.sim_file_handle is None:
        _open_file()

    fh  = st.session_state.sim_file_handle
    line = fh.readline()

    # -----------------------------------------------------------------------
    # EOF → loop back to the start
    # -----------------------------------------------------------------------
    if not line:
        fh.seek(0)
        st.session_state.sim_looped     = True
        st.session_state.sim_loop_count += 1
        line = fh.readline()
        if not line:
            # File is completely empty — nothing to do
            return None

    # Parse the CSV line
    try:
        row = next(csv.reader([line]))
        if len(row) < len(COLUMN_NAMES):
            # Skip malformed lines silently
            return read_next_record()
        return _coerce_row(row)
    except Exception:
        return read_next_record()
