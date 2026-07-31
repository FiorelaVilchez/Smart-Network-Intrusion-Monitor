"""Full integration test — no Streamlit needed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, datetime, pandas as pd
from src.storage import init_db, insert_record, get_last_n, get_total_count
from src.preprocessing import predict_record

with open("COLUMN_SCHEMA.json") as f:
    schema = json.load(f)

COLS = schema["columns"]
df = pd.read_csv("data/KDDTest+.txt", header=None, names=COLS, nrows=5)

init_db()

for i, row in df.iterrows():
    raw = row.to_dict()
    pred = predict_record(raw)
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    insert_record(ts, str(raw["label"]), pred, raw)
    tag = "ANOMALY" if pred["is_anomaly"] else "normal"
    print(f"Row {i}: label={raw['label']!r:20s} -> {tag}, score={pred['anomaly_score']:.4f}")

total = get_total_count()
print(f"\nTotal records in DB: {total}")
recent = get_last_n(5)
print(recent[["timestamp","ground_truth","is_anomaly","anomaly_score","protocol_type","service"]].to_string())
print("\nIntegration test PASSED!")
