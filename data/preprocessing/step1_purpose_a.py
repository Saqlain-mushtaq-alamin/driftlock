"""
DRIFTLOCK — Phase 1.3
Step 1: Purpose A — SOC Agent Task Input

Converts each row of NF-ToN-IoT-v2 into a natural-language classification
query that the Triage Agent can read and reason about.

Key design decisions (explained in comments):
  - IP addresses and ports are KEPT (agent needs them to reason semantically)
  - Outliers are NOT deleted (would remove entire attack categories)
  - StandardScaler is applied to numeric columns for normalisation
  - A 'packet asymmetry ratio' feature is added as it is a strong DDoS signal
  - Only a stratified sample is saved (17M rows is too large for daily use)

Usage:
    python step1_purpose_a.py --csv path/to/NF-ToN-IoT-v2.csv

Output:
    data/processed/purpose_a_agent_queries.csv
        Columns: query_id, label, agent_query, src_ip, dst_ip, is_attack
"""

import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from column_config import (
    LABEL_COL, SRC_IP_COL, DST_IP_COL, PROTO_COL,
    SRC_PORT_COL, DST_PORT_COL, DURATION_COL,
    BYTES_IN_COL, BYTES_OUT_COL, PKTS_IN_COL, PKTS_OUT_COL,
    TCP_FLAGS_COL, BENIGN_LABEL, ATTACK_CATEGORIES,
    PATH_PURPOSE_A, PATH_STATS, OUTPUT_DIR,
)

# ── How many query rows to produce per label (reduces file size) ──────────────
# 500 benign + 500 per attack category = ~4,500 rows total — plenty for agent
# experiments.  Increase if you want a larger evaluation set.
SAMPLE_PER_CLASS = 500

# ── Protocol number → human name mapping (from IANA) ─────────────────────────
PROTO_MAP = {
    "6": "TCP", "17": "UDP", "1": "ICMP", "58": "ICMPv6",
    "0": "HOPOPT", "2": "IGMP", "47": "GRE", "50": "ESP",
    "132": "SCTP",
}


def proto_name(val) -> str:
    """Return human-readable protocol name from numeric or string value."""
    return PROTO_MAP.get(str(int(float(val))), str(val)) if val else "Unknown"


def flag_name(val) -> str:
    """Convert TCP flag bitmask to readable string."""
    try:
        flags = int(float(val))
        names = []
        if flags & 0x01: names.append("FIN")
        if flags & 0x02: names.append("SYN")
        if flags & 0x04: names.append("RST")
        if flags & 0x08: names.append("PSH")
        if flags & 0x10: names.append("ACK")
        if flags & 0x20: names.append("URG")
        return "+".join(names) if names else "NONE"
    except Exception:
        return str(val)


def build_query(row) -> str:
    """
    Convert a single dataframe row into a natural-language query string.

    The format is designed to give the LLM agent enough context to reason
    about the traffic without overwhelming it.  Every field has a label so
    the model can parse it without counting positional tokens.
    """
    # Core identity fields
    src_ip   = str(row.get(SRC_IP_COL, "unknown"))
    dst_ip   = str(row.get(DST_IP_COL, "unknown"))
    proto    = proto_name(row.get(PROTO_COL, "0"))
    src_port = str(int(float(row.get(SRC_PORT_COL, 0))))
    dst_port = str(int(float(row.get(DST_PORT_COL, 0))))

    # Volume and timing
    duration_ms = float(row.get(DURATION_COL, 0))
    bytes_in    = int(float(row.get(BYTES_IN_COL, 0)))
    bytes_out   = int(float(row.get(BYTES_OUT_COL, 0)))
    pkts_in     = int(float(row.get(PKTS_IN_COL, 0)))
    pkts_out    = int(float(row.get(PKTS_OUT_COL, 0)))

    # Derived signals (computed here so the agent can reason about them)
    total_bytes   = bytes_in + bytes_out
    pkt_ratio     = round(pkts_out / pkts_in, 2) if pkts_in > 0 else 0
    avg_pkt_size  = round(total_bytes / (pkts_in + pkts_out), 1) \
                    if (pkts_in + pkts_out) > 0 else 0
    duration_sec  = round(duration_ms / 1000, 4)

    # Optional TCP flags
    flags_str = ""
    if TCP_FLAGS_COL and TCP_FLAGS_COL in row.index:
        flags_str = f"  TCP flags: {flag_name(row[TCP_FLAGS_COL])}."

    query = (
        f"Classify this NetFlow record:\n"
        f"  Source:      {src_ip}:{src_port}\n"
        f"  Destination: {dst_ip}:{dst_port}\n"
        f"  Protocol:    {proto}\n"
        f"  Duration:    {duration_sec} s\n"
        f"  Bytes in:    {bytes_in:,}    Bytes out: {bytes_out:,}    Total: {total_bytes:,}\n"
        f"  Packets in:  {pkts_in:,}    Packets out: {pkts_out:,}\n"
        f"  Packet asymmetry ratio (out/in): {pkt_ratio}\n"
        f"  Avg packet size: {avg_pkt_size} bytes.{flags_str}\n"
        f"\n"
        f"Is this traffic Benign or an attack? "
        f"If attack, specify the category: "
        f"dos, ransomware, scanning, xss, or password.\n"
        f"Provide your classification, confidence (high/medium/low), and reasoning."
    )
    return query


def run(csv_path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[Purpose A] Loading full dataset …")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")

    # ── Validate required columns exist ──────────────────────────────────────
    required = [LABEL_COL, SRC_IP_COL, DST_IP_COL, PROTO_COL,
                SRC_PORT_COL, DST_PORT_COL, DURATION_COL,
                BYTES_IN_COL, BYTES_OUT_COL, PKTS_IN_COL, PKTS_OUT_COL]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"\n  ERROR: These columns are missing from your CSV:\n  {missing_cols}")
        print("  Check column_config.py and update the column names to match "
              "what step0_inspect_dataset.py printed.")
        return

    # ── Fill NaN values ───────────────────────────────────────────────────────
    # Numeric columns: fill with 0 (missing values usually mean no traffic)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(0)

    # String columns: fill with "unknown"
    str_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df[str_cols] = df[str_cols].fillna("unknown")

    # ── NOTE: We do NOT delete outliers ──────────────────────────────────────
    # The original paper removes outliers via IQR/z-score, which eliminates
    # ~4M rows and can drop entire attack categories with low sample counts
    # (e.g., Ransomware and MITM have very few samples).
    # LLM agents reason from raw values, not ML-optimised features, so
    # outlier removal would make rare attack types invisible to the agent.
    print(f"  Outlier removal skipped intentionally — "
          f"rare attack categories preserved.")

    # ── Stratified sampling ───────────────────────────────────────────────────
    print(f"  Sampling up to {SAMPLE_PER_CLASS} rows per label …")
    frames = []
    label_stats = {}
    for lbl, grp in df.groupby(LABEL_COL):
        n = min(SAMPLE_PER_CLASS, len(grp))
        sampled = grp.sample(n=n, random_state=42)
        frames.append(sampled)
        label_stats[str(lbl)] = {"total_in_dataset": int(len(grp)),
                                  "sampled": int(n)}
    sample_df = pd.concat(frames, ignore_index=True).sample(
        frac=1, random_state=42).reset_index(drop=True)
    print(f"  Total sampled rows: {len(sample_df):,}")

    # ── StandardScaler on numeric columns (for reference/ML baselines) ────────
    # We scale a copy of the numeric columns and store them alongside the
    # raw values.  The agent query uses RAW values; scaled values are
    # available if you need them for an ML comparison baseline.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(sample_df[num_cols])
    scaled_df = pd.DataFrame(scaled, columns=[f"{c}_scaled" for c in num_cols])
    # Do not attach scaled columns to the query CSV — they are for ML baselines

    # ── Build natural-language queries ────────────────────────────────────────
    print("  Building natural-language agent queries …")
    queries = []
    for idx, row in sample_df.iterrows():
        queries.append(build_query(row))

    # ── Assemble output dataframe ─────────────────────────────────────────────
    out = pd.DataFrame({
        "query_id":  [f"q_{i:06d}" for i in range(len(sample_df))],
        "label":     sample_df[LABEL_COL].values,
        "is_attack": (sample_df[LABEL_COL] != BENIGN_LABEL).astype(int).values,
        "src_ip":    sample_df[SRC_IP_COL].values,
        "dst_ip":    sample_df[DST_IP_COL].values,
        "protocol":  sample_df[PROTO_COL].values,
        "src_port":  sample_df[SRC_PORT_COL].values,
        "dst_port":  sample_df[DST_PORT_COL].values,
        "agent_query": queries,
    })

    out.to_csv(PATH_PURPOSE_A, index=False)
    print(f"\n  Saved {len(out):,} queries → {PATH_PURPOSE_A}")

    # ── Save dataset stats for reference ─────────────────────────────────────
    stats = {
        "total_rows_in_dataset": int(len(df)),
        "total_columns": int(df.shape[1]),
        "label_distribution": label_stats,
        "benign_label": BENIGN_LABEL,
        "attack_categories_detected": list(label_stats.keys()),
    }
    with open(PATH_STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved dataset stats → {PATH_STATS}")

    # ── Print a sample query so you can verify the format ─────────────────────
    print("\n" + "─" * 60)
    print("SAMPLE QUERY (first row):")
    print("─" * 60)
    print(out["agent_query"].iloc[0])
    print("─" * 60)
    print(f"\nLabel for this sample: {out['label'].iloc[0]}")
    print("\n[Purpose A] Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Path to NF-ToN-IoT-v2.csv")
    args = parser.parse_args()
    run(args.csv)