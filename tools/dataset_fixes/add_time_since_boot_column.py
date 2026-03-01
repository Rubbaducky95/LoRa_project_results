"""
Add time_since_boot_ms column to all raw CSV files.
- For valid payload rows: extract first value from payload (ms since sender boot).
- For PACKET_LOST rows: use previous time_since_boot + tx_interval_ms.
- Keep payload unchanged.
"""
import csv
import os
import re

WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
RAW_ROOT = os.path.join(WORKSPACE, "raw_test_data")
CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")


def payload_first_value_ms(payload):
    """Extract first value from payload as ms since boot. Returns None if invalid."""
    if not payload or payload == "PACKET_LOST" or str(payload).strip().startswith("CFG "):
        return None
    parts = str(payload).strip('"').split(",")
    if not parts or not parts[0].strip():
        return None
    try:
        return float(parts[0].strip())
    except (ValueError, TypeError):
        return None


def process_file(path):
    """Add time_since_boot_ms column. Returns True if modified."""
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "payload" not in rows[0]:
        return False
    header = rows[0]
    if "time_since_boot_ms" in header:
        return False  # Already has column
    payload_idx = header.index("payload")
    tx_idx = header.index("tx_interval_ms") if "tx_interval_ms" in header else None
    # Insert time_since_boot_ms after payload
    new_header = header[: payload_idx + 1] + ["time_since_boot_ms"] + header[payload_idx + 1 :]
    prev_time_ms = None
    for i in range(1, len(rows)):
        r = rows[i]
        while len(r) < len(header):
            r.append("")
        payload = str(r[payload_idx]).strip()
        if payload.startswith("CFG "):
            r.insert(payload_idx + 1, "")
            continue
        t_ms = payload_first_value_ms(payload)
        if t_ms is not None:
            prev_time_ms = t_ms
            r.insert(payload_idx + 1, str(int(t_ms)))
        else:
            # PACKET_LOST: use prev + tx_interval
            tx_ms = 1550.0
            if tx_idx is not None and len(r) > tx_idx and r[tx_idx]:
                try:
                    tx_ms = float(r[tx_idx])
                except (ValueError, TypeError):
                    pass
            if prev_time_ms is not None:
                prev_time_ms = prev_time_ms + tx_ms
                r.insert(payload_idx + 1, str(int(prev_time_ms)))
            else:
                r.insert(payload_idx + 1, "")
    # Update header in rows
    rows[0] = new_header
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    return True


def main():
    count = 0
    for root, _, files in os.walk(RAW_ROOT):
        for fn in files:
            if CFG_RE.match(fn):
                path = os.path.join(root, fn)
                if process_file(path):
                    count += 1
                    rel = os.path.relpath(path, RAW_ROOT)
                    print(f"  Added time_since_boot_ms: {rel}")
    print(f"Done. Modified {count} files.")


if __name__ == "__main__":
    main()
