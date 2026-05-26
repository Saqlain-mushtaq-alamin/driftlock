 
"""
DRIFTLOCK — Phase 1.3
Step 0: Dataset Inspection
 
Run this FIRST before any other script.
It tells you exactly what is in your CSV so the rest of the
pipeline can be written correctly.
 
Usage:
    python step0_inspect_dataset.py --csv path/to/NF-ToN-IoT-v2.csv
    python data/preprocessing/step0_inspect_dataset.py --csv data/raw/NF-ToN-IoT-v2.csv
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os


def inspect(csv_path: str):
    print("=" * 65)
    print("DRIFTLOCK Dataset Inspector — NF-ToN-IoT-v2")
    print("=" * 65)

    # ── 1. Load (just the first 10k rows first to be fast) ──────────
    print(f"\n[1] Loading first 10,000 rows from:\n    {csv_path}\n")
    try:
        sample = pd.read_csv(csv_path, nrows=10_000, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: File not found at {csv_path}")
        sys.exit(1)

    # ── 2. Basic shape ───────────────────────────────────────────────
    print(f"[2] Sample shape: {sample.shape[0]} rows × {sample.shape[1]} columns\n")

    # ── 3. All column names ─────────────────────────────────────────
    print("[3] ALL COLUMN NAMES:")
    for i, col in enumerate(sample.columns, 1):
        print(f"    {i:>3}. {col}")

    # ── 4. Find the label column ────────────────────────────────────
    print("\n[4] SEARCHING FOR LABEL COLUMN:")
    label_candidates = [c for c in sample.columns
                        if c.lower() in ('label', 'attack', 'class',
                                         'attack_cat', 'category', 'type')]
    if label_candidates:
        for lc in label_candidates:
            print(f"    Found candidate: '{lc}'")
            print(f"    Unique values  : {sample[lc].unique().tolist()}")
    else:
        print("    No obvious label column found.")
        print("    Columns with low cardinality (likely categorical):")
        for c in sample.columns:
            if sample[c].nunique() < 20:
                print(f"      '{c}': {sample[c].unique().tolist()}")

    # ── 5. IP / port / timestamp columns ────────────────────────────
    print("\n[5] IP / PORT / TIMESTAMP COLUMNS (keep for agent queries):")
    keep_keywords = ['ip', 'port', 'addr', 'src', 'dst', 'time',
                     'stamp', 'proto', 'protocol', 'flow_id']
    found_keep = [c for c in sample.columns
                  if any(k in c.lower() for k in keep_keywords)]
    for c in found_keep:
        print(f"    {c}")

    # ── 6. Numeric feature columns ───────────────────────────────────
    num_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n[6] NUMERIC FEATURE COLUMNS ({len(num_cols)} total):")
    for c in num_cols:
        print(f"    {c}")

    # ── 7. Missing values ────────────────────────────────────────────
    missing = sample.isnull().sum()
    cols_with_missing = missing[missing > 0]
    print(f"\n[7] COLUMNS WITH MISSING VALUES: "
          f"{'None' if cols_with_missing.empty else ''}")
    if not cols_with_missing.empty:
        print(cols_with_missing.to_string())

    # ── 8. Load full file to get exact row count and distribution ────
    print("\n[8] LOADING FULL FILE FOR LABEL DISTRIBUTION ...")
    print("    (This may take 1–2 minutes for 17M rows)")
    full = pd.read_csv(csv_path, low_memory=False)
    print(f"\n    Full dataset shape: {full.shape[0]:,} rows × {full.shape[1]} columns")

    if label_candidates:
        label_col = label_candidates[0]
        dist = full[label_col].value_counts()
        total = len(full)
        print(f"\n    Label distribution (column='{label_col}'):")
        print(f"    {'Label':<30} {'Count':>10}  {'%':>6}")
        print(f"    {'-'*50}")
        for lbl, cnt in dist.items():
            print(f"    {str(lbl):<30} {cnt:>10,}  {cnt/total*100:>5.1f}%")
    else:
        print("\n    Could not auto-detect label column. See column list above.")

    # ── 9. Memory check ──────────────────────────────────────────────
    mem_mb = full.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"\n[9] MEMORY USAGE of full dataframe: {mem_mb:,.0f} MB")

    print("\n" + "=" * 65)
    print("NEXT STEP: Open step1_purpose_a.py and confirm the column")
    print("names match what you see above, then run it.")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Path to NF-ToN-IoT-v2.csv")
    args = parser.parse_args()
    inspect(args.csv)