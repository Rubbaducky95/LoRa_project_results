"""
Recalculate time_since_boot_ms for PACKET_LOST rows from previous row + tx_interval.
Use when PACKET_LOST times were incorrectly overwritten by fix_synthetic_timestamps.
Only processes files that have NO synthetic values (1000000-1010000).
"""
import csv
import os
import re

WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
RAW_ROOT = os.path.join(WORKSPACE, "raw_test_data")
CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")
SYNTHETIC_MIN, SYNTHETIC_MAX = 1000000, 1010000


def is_synthetic(val):
    try:
        v = float(val)
        return SYNTHETIC_MIN <= v <= SYNTHETIC_MAX
    except (ValueError, TypeError):
        return False


def file_has_synthetic(rows, header):
    time_idx = header.index("time_since_boot_ms") if "time_since_boot_ms" in header else None
    if time_idx is None:
        return False
    for r in rows[1:]:
        if time_idx < len(r) and r[time_idx] and is_synthetic(r[time_idx]):
            return True
    return False


def fix_file(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "payload" not in rows[0]:
        return False
    header = rows[0]
    if file_has_synthetic(rows, header):
        return False
    payload_idx = header.index("payload")
    time_idx = header.index("time_since_boot_ms") if "time_since_boot_ms" in header else None
    tx_idx = header.index("tx_interval_ms") if "tx_interval_ms" in header else None
    if time_idx is None:
        return False

    prev_time = None
    modified = False
    for i in range(1, len(rows)):
        r = rows[i]
        while len(r) < len(header):
            r.append("")
        payload = str(r[payload_idx]).strip()
        if payload.startswith("CFG "):
            continue
        if payload == "PACKET_LOST":
            tx_ms = 1550.0
            if tx_idx is not None and len(r) > tx_idx and r[tx_idx]:
                try:
                    tx_ms = float(r[tx_idx])
                except (ValueError, TypeError):
                    pass
            if prev_time is not None:
                new_ms = int(prev_time + tx_ms)
                if time_idx < len(r) and r[time_idx] != str(new_ms):
                    r[time_idx] = str(new_ms)
                    modified = True
                prev_time = new_ms
            continue
        if time_idx < len(r) and r[time_idx]:
            try:
                prev_time = float(r[time_idx])
            except (ValueError, TypeError):
                pass

    if modified:
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
    return modified


def main():
    count = 0
    for root, _, files in os.walk(RAW_ROOT):
        for fn in files:
            if CFG_RE.match(fn):
                path = os.path.join(root, fn)
                if fix_file(path):
                    count += 1
                    print(f"  Fixed PACKET_LOST: {os.path.relpath(path, RAW_ROOT)}")
    print(f"Done. Fixed {count} files.")


if __name__ == "__main__":
    main()
