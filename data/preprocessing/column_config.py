"""
DRIFTLOCK — Phase 1.3
column_config.py  — EDIT THIS FILE after running step0_inspect_dataset.py

After you run step0 you will see the exact column names printed.
Replace the values below with what you actually see.

NF-ToN-IoT-v2 typically uses these names — but confirm with step0 output.
"""

# ── The column that contains the ground-truth label ─────────────────────────
# Common values: 'Label', 'Attack', 'attack_cat', 'Category'
LABEL_COL = "Attack"

# ── Source IP column name ────────────────────────────────────────────────────
SRC_IP_COL = "IPV4_SRC_ADDR"

# ── Destination IP column name ───────────────────────────────────────────────
DST_IP_COL = "IPV4_DST_ADDR"

# ── Protocol column name ─────────────────────────────────────────────────────
PROTO_COL = "PROTOCOL"

# ── Source port column name ──────────────────────────────────────────────────
SRC_PORT_COL = "L4_SRC_PORT"

# ── Destination port column name ─────────────────────────────────────────────
DST_PORT_COL = "L4_DST_PORT"

# ── Flow duration column name ─────────────────────────────────────────────────
DURATION_COL = "FLOW_DURATION_MILLISECONDS"

# ── Bytes in / out columns ────────────────────────────────────────────────────
BYTES_IN_COL  = "IN_BYTES"
BYTES_OUT_COL = "OUT_BYTES"

# ── Packets in / out columns ──────────────────────────────────────────────────
PKTS_IN_COL  = "IN_PKTS"
PKTS_OUT_COL = "OUT_PKTS"

# ── TCP flags column (optional — set to None if not present) ─────────────────
TCP_FLAGS_COL = "TCP_FLAGS"   # or None

# ── Attack category labels in this dataset ───────────────────────────────────
# The exact strings that appear in LABEL_COL for attack traffic.
# The string for normal / benign traffic goes in BENIGN_LABEL.
BENIGN_LABEL = "Benign"   # or "0", "Normal", "BENIGN" — check step0 output

ATTACK_CATEGORIES = [
    "dos",
    "ddos",
    "ransomware",
    "scanning",
    "xss",
    "password",
    "backdoor",
    "injection",
    "mitm",
]
# If the dataset uses different strings (e.g. "DoS" instead of "DDoS"),
# update the list above to match the step0 output exactly.


# ── Output file paths ─────────────────────────────────────────────────────────
OUTPUT_DIR = "data/processed"

PATH_PURPOSE_A   = f"{OUTPUT_DIR}/purpose_a_agent_queries.csv"
PATH_PURPOSE_B   = f"{OUTPUT_DIR}/purpose_b_seed_corpus.json"
PATH_PURPOSE_C   = f"{OUTPUT_DIR}/purpose_c_minja_pairs.json"
PATH_STATS       = f"{OUTPUT_DIR}/dataset_stats.json"