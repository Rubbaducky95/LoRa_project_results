"""
Fix synthetic payload timestamps for distance_93.75m and distance_100.0m.
- 93.75m: Don't touch SF7_BW62500_TP2, SF7_BW62500_TP12. Start from last value in TP12 (415956),
  continue through remaining files in order: TP, BW, SF.
- 100.0m: Start from first value in 93.75m SF7_BW62500_TP2 (106652), continue through all files.
"""
import csv
import os
import re

WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
RAW_ROOT = os.path.join(WORKSPACE, "raw_test_data")

BW_ORDER = [62500, 125000, 250000, 500000]
TP_ORDER = [2, 12, 22]
SF_ORDER = [7, 8, 9, 10, 11, 12]

CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")


def get_file_order():
    """Yield (sf, bw, tp) in test order: TP first, then BW, then SF."""
    for sf in SF_ORDER:
        for bw in BW_ORDER:
            for tp in TP_ORDER:
                yield (sf, bw, tp)


def get_last_payload_value(path):
    """Get last valid payload first value from file. Returns (last_value, avg_tx_interval)."""
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "payload" not in rows[0]:
        return None, None
    header = rows[0]
    payload_idx = header.index("payload")
    tx_idx = header.index("tx_interval_ms") if "tx_interval_ms" in header else None
    last_val = None
    tx_vals = []
    for r in rows[1:]:
        if len(r) <= payload_idx:
            continue
        payload = str(r[payload_idx]).strip()
        if payload.startswith("CFG ") or payload == "PACKET_LOST":
            continue
        parts = payload.strip('"').split(",")
        if parts and parts[0].strip().isdigit():
            last_val = int(float(parts[0].strip()))
        if tx_idx is not None and len(r) > tx_idx and r[tx_idx]:
            try:
                tx_vals.append(float(r[tx_idx]))
            except (ValueError, TypeError):
                pass
    avg_tx = sum(tx_vals) / len(tx_vals) if tx_vals else 1550.0
    return last_val, avg_tx


def get_first_payload_value(path):
    """Get first valid payload first value from file."""
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "payload" not in rows[0]:
        return None
    header = rows[0]
    payload_idx = header.index("payload")
    for r in rows[1:]:
        if len(r) <= payload_idx:
            continue
        payload = str(r[payload_idx]).strip()
        if payload.startswith("CFG ") or payload == "PACKET_LOST":
            continue
        parts = payload.strip('"').split(",")
        if parts and parts[0].strip().replace(".", "").isdigit():
            return int(float(parts[0].strip()))
    return None


def fix_file(path, start_ms, use_tx_from_rows=True):
    """
    Rewrite payload first values to be sequential. start_ms is the first packet timestamp.
    Returns the last value written (for chaining).
    """
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "payload" not in rows[0]:
        return start_ms
    header = rows[0]
    payload_idx = header.index("payload")
    tx_idx = header.index("tx_interval_ms") if "tx_interval_ms" in header else None

    current_ms = start_ms
    for i in range(1, len(rows)):
        r = rows[i]
        while len(r) < len(header):
            r.append("")
        payload = str(r[payload_idx]).strip()
        if payload.startswith("CFG ") or payload == "PACKET_LOST":
            continue
        parts = payload.strip('"').split(",")
        if not parts:
            continue
        # Replace first part with current_ms
        tx_ms = 1550.0
        if use_tx_from_rows and tx_idx is not None and len(r) > tx_idx and r[tx_idx]:
            try:
                tx_ms = float(r[tx_idx])
            except (ValueError, TypeError):
                pass
        parts[0] = str(int(current_ms))
        r[payload_idx] = ",".join(parts)
        current_ms += tx_ms

    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    return int(current_ms)


def fix_quotes_in_file(path):
    """Remove extra CSV escaping from payload (e.g. \"\"\"417505,...\"\"\" -> \"417505,...\")."""
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "payload" not in rows[0]:
        return False
    header = rows[0]
    payload_idx = header.index("payload")
    changed = False
    for r in rows[1:]:
        while len(r) < len(header):
            r.append("")
        payload = str(r[payload_idx]).strip()
        if payload.startswith("CFG ") or payload == "PACKET_LOST":
            continue
        parts = payload.strip('"').split(",")
        if not parts:
            continue
        new_val = ",".join(parts)
        if r[payload_idx] != new_val:
            r[payload_idx] = new_val
            changed = True
    if changed:
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
    return changed


def find_file_in_dist(dist_path, sf, bw, tp):
    """Find CSV path for given config within a distance folder."""
    for sub in os.listdir(dist_path):
        spath = os.path.join(dist_path, sub)
        if not os.path.isdir(spath):
            continue
        for fn in os.listdir(spath):
            m = CFG_RE.match(fn)
            if m and tuple(map(int, m.groups())) == (sf, bw, tp):
                return os.path.join(spath, fn)
    return None


def main():
    # 93.75m: get last value from SF7_BW62500_TP12
    dist_9375 = os.path.join(RAW_ROOT, "distance_93.75m")
    tp12_path = find_file_in_dist(dist_9375, 7, 62500, 12) if os.path.isdir(dist_9375) else None
    if not tp12_path:
        tp12_path = os.path.join(dist_9375, "SF7", "SF7_BW62500_TP12.csv")
    if not os.path.isfile(tp12_path):
        print(f"Warning: {tp12_path} not found")
        start_9375 = 415956 + 1550  # fallback
    else:
        last_val, avg_tx = get_last_payload_value(tp12_path)
        start_9375 = (last_val or 415956) + int(avg_tx or 1550)
    print(f"93.75m: Starting from {start_9375} (after TP12)")

    # 100m: get first value from 93.75m SF7_BW62500_TP2
    tp2_9375_path = find_file_in_dist(dist_9375, 7, 62500, 2) if os.path.isdir(dist_9375) else None
    if not tp2_9375_path:
        tp2_9375_path = os.path.join(dist_9375, "SF7", "SF7_BW62500_TP2.csv")
    first_100 = get_first_payload_value(tp2_9375_path) if os.path.isfile(tp2_9375_path) else 106652
    print(f"100m: Starting from {first_100} (same as 93.75m TP2 first)")

    skip_9375 = {(7, 62500, 2), (7, 62500, 12)}

    # Process 93.75m
    if os.path.isdir(dist_9375):
        current = start_9375
        for sf, bw, tp in get_file_order():
            if (sf, bw, tp) in skip_9375:
                continue
            path = find_file_in_dist(dist_9375, sf, bw, tp)
            if path and os.path.isfile(path):
                # Check if file has synthetic data (1000001-1000100)
                first_val = get_first_payload_value(path)
                if first_val and 1000000 <= first_val <= 1000200:
                    last = fix_file(path, current)
                    print(f"  Fixed 93.75 SF{sf} BW{bw} TP{tp}: {current} -> {last}")
                    current = last

    # Process 100m - all files
    dist_100 = os.path.join(RAW_ROOT, "distance_100.0m")
    if os.path.isdir(dist_100):
        current = first_100
        for sf, bw, tp in get_file_order():
            path = find_file_in_dist(dist_100, sf, bw, tp)
            if path and os.path.isfile(path):
                first_val = get_first_payload_value(path)
                if first_val and 1000000 <= first_val <= 1000200:
                    last = fix_file(path, current)
                    print(f"  Fixed 100 SF{sf} BW{bw} TP{tp}: {current} -> {last}")
                    current = last

    # Fix extra quotes in any files that have \"\"\"...\"\"\" format (skip real 93.75 TP2/TP12)
    print("Fixing payload quote format...")
    for dist_path in [dist_9375, dist_100]:
        if not os.path.isdir(dist_path):
            continue
        for sf, bw, tp in get_file_order():
            if dist_path == dist_9375 and (sf, bw, tp) in skip_9375:
                continue  # Don't touch real test data
            path = find_file_in_dist(dist_path, sf, bw, tp)
            if path and os.path.isfile(path) and fix_quotes_in_file(path):
                print(f"  Fixed quotes: {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}")

    print("Done.")


if __name__ == "__main__":
    main()
