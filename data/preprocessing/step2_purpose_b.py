"""
DRIFTLOCK — Phase 1.3
Step 2: Purpose B — Seed Corpus for Layer 2 (Domain-Anchored Bootstrap)

Selects high-confidence, unambiguous records from each attack category
and converts them into natural-language memory entries that will be
pre-loaded into the agent's memory store before any live traffic.

This is the dataset preparation for DRIFTLOCK Layer 2 (DAB).

What "high-confidence / unambiguous" means per category:
  - ddos      : very high OUT_PKTS, very short duration, many small packets
  - dos       : high IN_BYTES or OUT_BYTES, short duration, high pkt rate
  - scanning  : many unique destination ports, small packet sizes, SYN flags
  - ransomware: large OUT_BYTES, long duration, few source IPs, unusual ports
  - xss       : HTTP-like port (80/443/8080), small payloads, many requests
  - password  : port 22 (SSH) or 21 (FTP) or 23 (Telnet), repeated attempts
  - backdoor  : unusual high dst port, persistent connection, bidirectional
  - injection : web ports (80/443), moderate payload, repeated src->dst
  - mitm      : very symmetric in/out bytes and packets (interceptor pattern)
  - Benign    : normal traffic, balanced bytes, standard ports

Each record is saved as a memory entry with:
  - A natural-language description of what the traffic looks like
  - The correct classification label
  - A human-readable explanation of WHY it is that category
  - The raw provenance metadata (used by DRIFTLOCK Layer 1 for signing)

Usage:
    python step2_purpose_b.py --csv path/to/NF-ToN-IoT-v2.csv

Output:
    data/processed/purpose_b_seed_corpus.json
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from column_config import (
    LABEL_COL, SRC_IP_COL, DST_IP_COL, PROTO_COL,
    SRC_PORT_COL, DST_PORT_COL, DURATION_COL,
    BYTES_IN_COL, BYTES_OUT_COL, PKTS_IN_COL, PKTS_OUT_COL,
    TCP_FLAGS_COL, BENIGN_LABEL, ATTACK_CATEGORIES,
    PATH_PURPOSE_B, OUTPUT_DIR,
)

# ── How many seed entries to select per category ──────────────────────────────
# 3 per category × 10 categories = 30 total seed entries.
# This is above the minimum of 20 recommended in the DRIFTLOCK methodology.
SEEDS_PER_CLASS = 3

# ── Scoring functions — one per attack category ───────────────────────────────
# Each function takes a DataFrame of rows from one category and returns a
# "confidence score" (higher = more unambiguous example of that category).
# The top SEEDS_PER_CLASS rows by score are selected as seed entries.

def score_ddos(df: pd.DataFrame) -> pd.Series:
    """
    DDoS: massive outgoing packet flood in very short time.
    Signals: high OUT_PKTS, high pkt asymmetry (out >> in), short duration,
             high SRC_TO_DST_AVG_THROUGHPUT.
    """
    pkt_ratio  = df[PKTS_OUT_COL] / (df[PKTS_IN_COL] + 1)
    throughput = (
    df["SRC_TO_DST_AVG_THROUGHPUT"]
    if "SRC_TO_DST_AVG_THROUGHPUT" in df.columns
    else pd.Series(0.0, index=df.index)
)

    short_dur  = 1 / (df[DURATION_COL] + 1)   # shorter = higher score
    return (
        _norm(df[PKTS_OUT_COL]) * 0.35 +
        _norm(pkt_ratio)         * 0.30 +
        _norm(throughput)        * 0.20 +
        _norm(short_dur)         * 0.15
    )

def score_dos(df: pd.DataFrame) -> pd.Series:
    """
    DoS: volumetric attack — high total bytes, high packet rate.
    """
    total_bytes = df[BYTES_IN_COL] + df[BYTES_OUT_COL]
    total_pkts  = df[PKTS_IN_COL]  + df[PKTS_OUT_COL]
    pkt_rate    = total_pkts / (df[DURATION_COL] + 1)
    return (
        _norm(total_bytes) * 0.40 +
        _norm(total_pkts)  * 0.30 +
        _norm(pkt_rate)    * 0.30
    )

def score_scanning(df: pd.DataFrame) -> pd.Series:
    """
    Scanning: small packets, SYN flag pattern, low bytes per packet.
    Proxy: very short duration, very small OUT_BYTES, SYN flags.
    """
    small_payload = 1 / (df[BYTES_OUT_COL] + 1)   # smaller = more scan-like
    short_dur     = 1 / (df[DURATION_COL] + 1)
    small_pkts = (
    1 / (df["LONGEST_FLOW_PKT"] + 1)
    if "LONGEST_FLOW_PKT" in df.columns
    else pd.Series(0.0, index=df.index)
)
    return (
        _norm(small_payload) * 0.40 +
        _norm(short_dur)     * 0.35 +
        _norm(small_pkts)    * 0.25
    )

def score_ransomware(df: pd.DataFrame) -> pd.Series:
    """
    Ransomware: large outgoing data (exfiltration), long duration,
    unusual destination ports (not 80/443).
    """
    large_out  = df[BYTES_OUT_COL]
    long_dur   = df[DURATION_COL]
    unusual_port = (~df[DST_PORT_COL].isin([80, 443, 22, 53, 8080])).astype(float)
    return (
        _norm(large_out)    * 0.40 +
        _norm(long_dur)     * 0.35 +
        _norm(unusual_port) * 0.25
    )

def score_xss(df: pd.DataFrame) -> pd.Series:
    """
    XSS: web ports (80, 443, 8080), moderate payload, many packets.
    """
    web_port   = df[DST_PORT_COL].isin([80, 443, 8080, 8000, 8443]).astype(float)
    total_pkts = df[PKTS_IN_COL] + df[PKTS_OUT_COL]
    return (
        _norm(web_port)    * 0.50 +
        _norm(total_pkts)  * 0.30 +
        _norm(df[BYTES_IN_COL]) * 0.20
    )

def score_password(df: pd.DataFrame) -> pd.Series:
    """
    Password attack (brute force): SSH (22), FTP (21), Telnet (23),
    RDP (3389). Many small repeated packets.
    """
    auth_port  = df[DST_PORT_COL].isin([22, 21, 23, 3389, 445]).astype(float)
    many_pkts  = df[PKTS_IN_COL] + df[PKTS_OUT_COL]
    return (
        _norm(auth_port)  * 0.55 +
        _norm(many_pkts)  * 0.45
    )

def score_backdoor(df: pd.DataFrame) -> pd.Series:
    """
    Backdoor: persistent bidirectional connection, unusual high dst port,
    relatively balanced in/out bytes.
    """
    balance    = 1 / (abs(df[BYTES_IN_COL] - df[BYTES_OUT_COL]) + 1)
    long_dur   = df[DURATION_COL]
    high_port  = (df[DST_PORT_COL] > 1024).astype(float)
    return (
        _norm(balance)   * 0.40 +
        _norm(long_dur)  * 0.35 +
        _norm(high_port) * 0.25
    )

def score_injection(df: pd.DataFrame) -> pd.Series:
    """
    Injection (SQL/command): web ports, moderate payload, repeated src->dst.
    Similar to XSS but with larger individual packet sizes.
    """
    web_port  = df[DST_PORT_COL].isin([80, 443, 8080, 3306, 5432]).astype(float)
    pkt_size  = (df[BYTES_IN_COL] + df[BYTES_OUT_COL]) / \
                (df[PKTS_IN_COL]  + df[PKTS_OUT_COL]  + 1)
    return (
        _norm(web_port) * 0.50 +
        _norm(pkt_size) * 0.50
    )

def score_mitm(df: pd.DataFrame) -> pd.Series:
    """
    MITM: very symmetric in/out — the attacker is relaying traffic.
    Near-equal bytes and packets in both directions.
    """
    byte_sym = 1 / (abs(df[BYTES_IN_COL] - df[BYTES_OUT_COL]) + 1)
    pkt_sym  = 1 / (abs(df[PKTS_IN_COL]  - df[PKTS_OUT_COL])  + 1)
    return (
        _norm(byte_sym) * 0.50 +
        _norm(pkt_sym)  * 0.50
    )

def score_benign(df: pd.DataFrame) -> pd.Series:
    """
    Benign: well-formed TCP/UDP on standard ports, normal byte volumes.
    Pick records that are the most "ordinary" looking.
    """
    std_port   = df[DST_PORT_COL].isin([80, 443, 53, 22, 25, 110, 143,
                                         8080, 3306]).astype(float)
    normal_vol = 1 / (df[BYTES_OUT_COL] + df[BYTES_IN_COL] + 1)
    return (
        _norm(std_port)   * 0.50 +
        _norm(normal_vol) * 0.50
    )

# ── Score dispatcher ──────────────────────────────────────────────────────────
SCORE_FN = {
    "ddos":       score_ddos,
    "dos":        score_dos,
    "scanning":   score_scanning,
    "ransomware": score_ransomware,
    "xss":        score_xss,
    "password":   score_password,
    "backdoor":   score_backdoor,
    "injection":  score_injection,
    "mitm":       score_mitm,
    BENIGN_LABEL: score_benign,
}

# ── Helper ────────────────────────────────────────────────────────────────────
def _norm(series: pd.Series) -> pd.Series:
    """Min-max normalise a series to [0, 1]. Returns zeros if constant."""
    if isinstance(series, (int, float)):
        return pd.Series([series] * 1, dtype=float)
    rng = series.max() - series.min()
    if rng == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / rng

PROTO_MAP = {6: "TCP", 17: "UDP", 1: "ICMP", 58: "ICMPv6",
             0: "HOPOPT", 2: "IGMP", 47: "GRE", 50: "ESP", 132: "SCTP"}

def proto_name(val) -> str:
    try:
        return PROTO_MAP.get(int(float(val)), str(val))
    except Exception:
        return str(val)

def flag_name(val) -> str:
    try:
        f = int(float(val))
        names = []
        if f & 0x01: names.append("FIN")
        if f & 0x02: names.append("SYN")
        if f & 0x04: names.append("RST")
        if f & 0x08: names.append("PSH")
        if f & 0x10: names.append("ACK")
        if f & 0x20: names.append("URG")
        return "+".join(names) if names else "NONE"
    except Exception:
        return str(val)

# ── Explanation templates per category ────────────────────────────────────────
# These sentences appear in the memory entry and tell the agent WHY this
# record was classified as that category — this is what makes it useful
# as a demonstration during future inference.
EXPLANATIONS = {
    "ddos": (
        "This is a DDoS attack: the packet asymmetry ratio (outgoing >> incoming) "
        "combined with extremely short duration and very high outgoing packet count "
        "is the signature of a volumetric flooding attack overwhelming the target."
    ),
    "dos": (
        "This is a DoS attack: the total byte volume is abnormally high for the "
        "connection duration, and the packet rate indicates a deliberate attempt "
        "to exhaust the target's resources."
    ),
    "scanning": (
        "This is a Scanning attack: the very small outgoing payload, minimal "
        "duration, and SYN-only or minimal TCP flag pattern are consistent with "
        "a port scanner probing the target for open services."
    ),
    "ransomware": (
        "This is Ransomware: the large outgoing byte volume on an unusual destination "
        "port over an extended duration is consistent with ransomware exfiltrating "
        "data or communicating with a command-and-control server before encryption."
    ),
    "xss": (
        "This is an XSS attack: the traffic targets web ports (80/443/8080) with "
        "repeated small requests, consistent with an attacker injecting malicious "
        "script payloads into web application endpoints."
    ),
    "password": (
        "This is a Password (brute force) attack: the traffic targets authentication "
        "ports (SSH=22, FTP=21, Telnet=23, RDP=3389) with many repeated small "
        "packet exchanges, consistent with credential stuffing or brute forcing."
    ),
    "backdoor": (
        "This is a Backdoor connection: the persistent bidirectional connection "
        "with balanced in/out bytes on an unusual high port is consistent with "
        "a backdoor shell maintaining command-and-control access."
    ),
    "injection": (
        "This is an Injection attack (SQL/command): web port traffic with larger "
        "average packet sizes than typical browsing, consistent with malicious "
        "payloads being sent to a database or application endpoint."
    ),
    "mitm": (
        "This is a MITM (Man-in-the-Middle) attack: the near-perfect symmetry "
        "between incoming and outgoing bytes and packets indicates the attacker "
        "is transparently relaying and intercepting traffic between two endpoints."
    ),
    BENIGN_LABEL: (
        "This is Benign traffic: standard port, normal byte volume, balanced "
        "packet sizes, and no anomalous asymmetry — consistent with regular "
        "user or service communication."
    ),
}


def build_memory_entry(row, label: str, entry_index: int) -> dict:
    """
    Convert a single high-confidence row into a memory entry dict.

    The 'memory_text' field is what the agent reads during retrieval.
    The 'metadata' field contains the provenance data for DRIFTLOCK Layer 1.
    """
    src_ip   = str(row.get(SRC_IP_COL, "unknown"))
    dst_ip   = str(row.get(DST_IP_COL, "unknown"))
    proto    = proto_name(row.get(PROTO_COL, 0))
    src_port = int(float(row.get(SRC_PORT_COL, 0)))
    dst_port = int(float(row.get(DST_PORT_COL, 0)))
    dur_ms   = float(row.get(DURATION_COL, 0))
    dur_s    = round(dur_ms / 1000, 4)
    b_in     = int(float(row.get(BYTES_IN_COL,  0)))
    b_out    = int(float(row.get(BYTES_OUT_COL, 0)))
    p_in     = int(float(row.get(PKTS_IN_COL,   0)))
    p_out    = int(float(row.get(PKTS_OUT_COL,  0)))
    ratio    = round(p_out / p_in, 2) if p_in > 0 else 0
    avg_pkt  = round((b_in + b_out) / (p_in + p_out), 1) \
               if (p_in + p_out) > 0 else 0

    flags_str = ""
    if TCP_FLAGS_COL and TCP_FLAGS_COL in row.index:
        flags_str = f" | TCP flags: {flag_name(row[TCP_FLAGS_COL])}"

    explanation = EXPLANATIONS.get(label, f"Classified as {label}.")

    memory_text = (
        f"[VERIFIED SEED MEMORY — {label.upper()}]\n"
        f"Past classification: {src_ip}:{src_port} → {dst_ip}:{dst_port} "
        f"via {proto}, duration {dur_s}s, "
        f"bytes_in={b_in:,} bytes_out={b_out:,}, "
        f"pkts_in={p_in} pkts_out={p_out}, "
        f"pkt_ratio={ratio}, avg_pkt={avg_pkt}B{flags_str}.\n"
        f"CLASSIFICATION: {label}\n"
        f"CONFIDENCE: high\n"
        f"REASONING: {explanation}"
    )

    metadata = {
        "entry_id":     f"seed_{label}_{entry_index:03d}",
        "label":        label,
        "is_attack":    label != BENIGN_LABEL,
        "is_seed":      True,
        "source":       "dataset_seed_NF-ToN-IoT-v2",
        # Provenance fields for DRIFTLOCK Layer 1 HMAC signing
        "src_ip":       src_ip,
        "dst_ip":       dst_ip,
        "proto":        proto,
        "src_port":     src_port,
        "dst_port":     dst_port,
        "timestamp":    "2024-01-01T00:00:00Z",  # fixed for reproducibility
        "duration_ms":  dur_ms,
        "bytes_in":     b_in,
        "bytes_out":    b_out,
        "pkts_in":      p_in,
        "pkts_out":     p_out,
    }

    return {"memory_text": memory_text, "metadata": metadata}


def run(csv_path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("DRIFTLOCK — Step 2: Purpose B — Seed Corpus Builder")
    print("=" * 65)

    # ── All labels to process ────────────────────────────────────────────────
    all_labels = [BENIGN_LABEL] + ATTACK_CATEGORIES
    print(f"\nLabels to process: {all_labels}")
    print(f"Seeds per class  : {SEEDS_PER_CLASS}")
    print(f"Total seed entries planned: {len(all_labels) * SEEDS_PER_CLASS}\n")

    # ── Load full dataset ────────────────────────────────────────────────────
    print("[1] Loading full dataset (this takes ~30–60 seconds) …")
    df = pd.read_csv(csv_path, low_memory=False)
    df[PKTS_IN_COL]  = pd.to_numeric(df[PKTS_IN_COL],  errors="coerce").fillna(0)
    df[PKTS_OUT_COL] = pd.to_numeric(df[PKTS_OUT_COL], errors="coerce").fillna(0)
    df[BYTES_IN_COL] = pd.to_numeric(df[BYTES_IN_COL], errors="coerce").fillna(0)
    df[BYTES_OUT_COL]= pd.to_numeric(df[BYTES_OUT_COL],errors="coerce").fillna(0)
    df[DURATION_COL] = pd.to_numeric(df[DURATION_COL], errors="coerce").fillna(0)
    df[DST_PORT_COL] = pd.to_numeric(df[DST_PORT_COL], errors="coerce").fillna(0)
    print(f"   Loaded {len(df):,} rows\n")

    # ── Process each label ───────────────────────────────────────────────────
    all_seed_entries = []

    for label in all_labels:
        label_df = df[df[LABEL_COL] == label].copy()

        if label_df.empty:
            print(f"  [{label:<12}] WARNING: No rows found — "
                  f"check LABEL_COL and ATTACK_CATEGORIES in column_config.py")
            continue

        # Get the scoring function for this category
        score_fn = SCORE_FN.get(label)
        if score_fn is None:
            print(f"  [{label:<12}] No scoring function defined — "
                  f"using random sample")
            top_rows = label_df.sample(n=min(SEEDS_PER_CLASS, len(label_df)),
                                        random_state=42)
        else:
            # Score all rows in this category and take the top N
            scores = score_fn(label_df)
            top_idx = scores.nlargest(SEEDS_PER_CLASS).index
            top_rows = label_df.loc[top_idx]

        # Build memory entries for the selected rows
        entries_for_label = []
        for i, (_, row) in enumerate(top_rows.iterrows()):
            entry = build_memory_entry(row, label, i)
            entries_for_label.append(entry)
            all_seed_entries.append(entry)

        print(f"  [{label:<12}] Selected {len(entries_for_label)} seed entries")

        # Print the first entry for this label so you can verify it looks good
        if entries_for_label:
            first_text = entries_for_label[0]["memory_text"]
            preview = first_text[:220].replace("\n", " ")
            print(f"               Preview: {preview}…\n")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "description": (
            "DRIFTLOCK Layer 2 (DAB) Seed Corpus — "
            "pre-verified memory entries for agent bootstrap. "
            "Each entry is HMAC-signed by layer1_wtpg.py before being "
            "written to the vector store."
        ),
        "total_entries": len(all_seed_entries),
        "seeds_per_class": SEEDS_PER_CLASS,
        "source_dataset": "NF-ToN-IoT-v2",
        "entries": all_seed_entries,
    }

    with open(PATH_PURPOSE_B, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=" * 65)
    print(f"DONE — {len(all_seed_entries)} seed entries saved")
    print(f"Output: {PATH_PURPOSE_B}")
    print("=" * 65)
    print("\nNEXT STEP: run step3_purpose_c.py to build MINJA attack pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Path to NF-ToN-IoT-v2.csv")
    args = parser.parse_args()
    run(args.csv)