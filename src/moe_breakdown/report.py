"""
Report writer -- combines breakdown + chart + raw events into a
reproducible artifact bundle.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .categorize import Breakdown, BUCKETS, BUCKET_STYLE


def write(breakdown: Breakdown, metadata: dict, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Machine-readable JSON report
    report = {
        "metadata": metadata,
        "summary": breakdown.as_dict(),
        "buckets": [
            {
                "id": b,
                "label": BUCKET_STYLE[b]["label"],
                "color": BUCKET_STYLE[b]["color"],
                "time_us": breakdown.per_bucket_us.get(b, 0.0),
                "time_ms": breakdown.per_bucket_us.get(b, 0.0) / 1000.0,
                "percent": round(breakdown.percent(b), 3),
                "count": breakdown.per_bucket_count.get(b, 0),
            }
            for b in BUCKETS
        ],
    }
    json_path = out_dir / "breakdown.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # 2) CSV (one row per bucket) -- easy to diff across runs
    csv_path = out_dir / "breakdown.csv"
    with open(csv_path, "w") as f:
        f.write("bucket,label,time_us,time_ms,percent,count\n")
        for b in BUCKETS:
            f.write(
                f"{b},{BUCKET_STYLE[b]['label']!s},"
                f"{breakdown.per_bucket_us.get(b, 0.0):.2f},"
                f"{breakdown.per_bucket_us.get(b, 0.0)/1000.0:.4f},"
                f"{breakdown.percent(b):.3f},"
                f"{breakdown.per_bucket_count.get(b, 0)}\n"
            )

    # 3) Per-event detail (so users can grep / re-bucket manually)
    events_path = out_dir / "events.jsonl"
    with open(events_path, "w") as f:
        for ev in breakdown.events:
            f.write(json.dumps(asdict(ev)) + "\n")

    return {"report": str(json_path), "csv": str(csv_path), "events": str(events_path)}
