#!/usr/bin/env python3
"""
Backfill derived columns into every results.csv found inside all_runs.zip.
Only adds columns that are not already present (idempotent).

Usage:
    python scripts/backfill_derived_columns.py [--zip-in PATH] [--zip-out PATH]

Defaults:
    --zip-in  all_runs.zip
    --zip-out  (same as --zip-in, i.e. overwrite)
"""

import argparse
import csv
import fnmatch
import io
import os
import shutil
import tempfile
import zipfile

# ---------------------------------------------------------------------------
# H100 SXM5 hardware constants
# ---------------------------------------------------------------------------
H100_TFLOPS_BF16 = 989.0
H100_HBM_BW_TBYPS = 3.35
H100_FLOPS_PER_BYTE = H100_TFLOPS_BF16 / H100_HBM_BW_TBYPS  # ~295.2
BYTES_PER_BF16 = 2

# Ordered list of derived column names — order matters for insertion into CSV.
DERIVED_COLUMNS = [
    "expert_weight_bytes_per_gpu",
    "dispatch_bytes_theoretical",
    "allgather_send_bytes",
    "allgather_recv_bytes",
    "tokens_per_expert_avg",
    "flops_per_expert",
    "expert_GEMM_flops",
    "expert_GEMM_AMI",
    "compute_bound_predicted",
    "imbalance_ratio_alpha",
]


def compute_derived(row: dict) -> dict:
    """Return a dict of all derived column values for a single CSV row."""
    ep = int(row["ep"])
    tp = int(row["tp"])
    tokens = int(row["tokens"])
    hidden_size = int(row["hidden_size"])
    intermediate_size = int(row["intermediate_size"])
    num_experts = int(row["num_experts"])
    topk = int(row["topk"])

    num_local_experts = num_experts // ep

    # Weight bytes on this GPU
    w1_bytes = num_local_experts * 2 * intermediate_size * hidden_size * BYTES_PER_BF16 // tp
    w2_bytes = num_local_experts * intermediate_size * hidden_size * BYTES_PER_BF16 // tp
    expert_weight_bytes_per_gpu = w1_bytes + w2_bytes

    # Dispatch bytes (allgather input)
    dispatch_bytes_theoretical = tokens * hidden_size * BYTES_PER_BF16 * (ep - 1) / ep

    # Allgather network bytes
    allgather_send_bytes = tokens * hidden_size * BYTES_PER_BF16
    allgather_recv_bytes = tokens * hidden_size * BYTES_PER_BF16 * (ep - 1)

    # GEMM flops and arithmetic intensity
    tokens_per_expert_avg = tokens * topk / num_experts
    flops_per_expert = 2 * tokens_per_expert_avg * (2 * intermediate_size * hidden_size)
    expert_GEMM_flops = flops_per_expert * num_local_experts
    expert_GEMM_AMI = (
        expert_GEMM_flops / expert_weight_bytes_per_gpu
        if expert_weight_bytes_per_gpu > 0
        else 0.0
    )

    # Roofline prediction
    compute_bound_predicted = int(expert_GEMM_AMI >= H100_FLOPS_PER_BYTE)

    # Imbalance ratio
    routing_mode = row.get("routing_mode", "")
    alpha_observed_raw = float(row.get("alpha_observed", 1.0))
    if routing_mode == "worst-case":
        imbalance_ratio_alpha = float(num_experts) / float(topk)
    else:
        imbalance_ratio_alpha = alpha_observed_raw

    return {
        "expert_weight_bytes_per_gpu": expert_weight_bytes_per_gpu,
        "dispatch_bytes_theoretical": dispatch_bytes_theoretical,
        "allgather_send_bytes": allgather_send_bytes,
        "allgather_recv_bytes": allgather_recv_bytes,
        "tokens_per_expert_avg": tokens_per_expert_avg,
        "flops_per_expert": flops_per_expert,
        "expert_GEMM_flops": expert_GEMM_flops,
        "expert_GEMM_AMI": expert_GEMM_AMI,
        "compute_bound_predicted": compute_bound_predicted,
        "imbalance_ratio_alpha": imbalance_ratio_alpha,
    }


def process_csv(raw_bytes: bytes) -> tuple[bytes, int, list[str]]:
    """
    Add missing derived columns to a CSV given as raw bytes.

    Returns:
        (updated_bytes, row_count, list_of_added_column_names)
    """
    text = raw_bytes.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    existing_cols = set(reader.fieldnames or [])

    # Determine which derived columns are missing
    cols_to_add = [c for c in DERIVED_COLUMNS if c not in existing_cols]

    rows = list(reader)
    original_fieldnames = list(reader.fieldnames or [])

    if not cols_to_add:
        return raw_bytes, len(rows), []

    new_fieldnames = original_fieldnames + cols_to_add

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=new_fieldnames,
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()

    for row in rows:
        derived = compute_derived(row)
        for col in cols_to_add:
            row[col] = derived[col]
        writer.writerow(row)

    return output.getvalue().encode("utf-8"), len(rows), cols_to_add


def backfill(zip_in: str, zip_out: str) -> None:
    updated_files: list[tuple[str, int, list[str]]] = []  # (name, rows, added_cols)

    # We'll build a new zip in a temp file, then move it into place.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip", dir=os.path.dirname(os.path.abspath(zip_out)))
    try:
        with zipfile.ZipFile(zip_in, "r") as zin, zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                # Skip directory entries (their filename ends with '/')
                if item.filename.endswith("/"):
                    zout.mkdir(item.filename.rstrip("/"))
                    continue

                raw = zin.read(item.filename)

                if fnmatch.fnmatch(item.filename, "*/results.csv"):
                    new_raw, row_count, added_cols = process_csv(raw)
                    zout.writestr(item, new_raw)
                    updated_files.append((item.filename, row_count, added_cols))
                else:
                    zout.writestr(item, raw)

        # Atomically replace the output zip
        shutil.move(tmp_path, zip_out)
        tmp_path = None  # prevent cleanup below
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Print summary
    print(f"\nBackfill complete: {zip_out}")
    print(f"{'CSV file':<70}  {'rows':>5}  {'cols added'}")
    print("-" * 110)
    total_modified = 0
    for name, rows, added in sorted(updated_files):
        if added:
            total_modified += 1
            print(f"{name:<70}  {rows:>5}  {', '.join(added)}")
        else:
            print(f"{name:<70}  {rows:>5}  (no changes)")
    print("-" * 110)
    print(f"Modified {total_modified} / {len(updated_files)} CSVs.\n")


def main():
    parser = argparse.ArgumentParser(description="Backfill derived columns into results CSVs inside a zip.")
    parser.add_argument("--zip-in", default="all_runs.zip", help="Input zip file (default: all_runs.zip)")
    parser.add_argument("--zip-out", default=None, help="Output zip file (default: overwrite --zip-in)")
    args = parser.parse_args()

    zip_in = args.zip_in
    zip_out = args.zip_out if args.zip_out is not None else zip_in

    if not os.path.isfile(zip_in):
        raise FileNotFoundError(f"Input zip not found: {zip_in}")

    backfill(zip_in, zip_out)


if __name__ == "__main__":
    main()
