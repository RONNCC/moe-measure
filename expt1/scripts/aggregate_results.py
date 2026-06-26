#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Aggregate per-parallel-point CSV files into one study CSV")
    p.add_argument("--study-root", required=True, help="e.g. runs/fused-moe-characterization")
    p.add_argument("--out", default=None, help="Default: <study-root>/aggregate.csv")
    args = p.parse_args(argv)

    study_root = Path(args.study_root)
    # Dirs are named either tp{tp}-ep{ep} (legacy) or {job_id}_tp{tp}-ep{ep}
    csv_paths = sorted(study_root.glob("*tp*-ep*/results.csv"))
    if not csv_paths:
        raise SystemExit(f"No results.csv files found under {study_root}")

    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in csv_paths:
        with path.open() as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                rows.append(row)

    out_csv = Path(args.out or study_root / "aggregate.csv")
    out_json = out_csv.with_suffix(".json")
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(json.dumps({"rows": rows, "num_rows": len(rows)}, indent=2))
    print(f"[aggregate] wrote {out_csv}")
    print(f"[aggregate] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
