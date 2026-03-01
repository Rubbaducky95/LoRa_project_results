"""
Fix synthetic timestamps (1000001, 1000002, ...) across all FILLED_ROWS and GENERATED_FROM_RAW files.
- Iterate through files in order (distance, then config: SF->BW->TP).
- When we see 1000001 and previous value was NOT (last_time + tx_interval): fix by continuing from previous + offset.
- On distance change: reboot, so 1000001 is valid (don't chain from previous distance).
- On dip (time went backwards): config restart, use previous + couple seconds as new start.
- Only replace values in synthetic range (1000000-1010000); never overwrite real values.
- Update both payload and time_since_boot_ms.
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
SYNTHETIC_MIN, SYNTHETIC_MAX = 1000000, 1010000
COUPLE_SECONDS_MS = 2500  # ~2.5 seconds between config changes


def parse_distance(name):
    m = re.match(r"^distance_([\d.]+)m?$", name)
    return float(m.group(1)) if m else None


def get_file_order():
    for sf in SF_ORDER:
        for bw in BW_ORDER:
            for tp in TP_ORDER:
                yield (sf, bw, tp)


def find_file_in_dist(dist_path, sf, bw, tp):
    for sub in os.listdir(dist_path):
        spath = os.path.join(dist_path, sub)
        if not os.path.isdir(spath):
            continue
        for fn in os.listdir(spath):
            m = CFG_RE.match(fn)
            if m and tuple(map(int, m.groups())) == (sf, bw, tp):
                return os.path.join(spath, fn)
    return None


def is_synthetic(val):
    try:
        v = float(val)
        return SYNTHETIC_MIN <= v <= SYNTHETIC_MAX
    except (ValueError, TypeError):
        return False


def file_has_synthetic(rows, header):
    """Check if file has any synthetic time_since_boot_ms values."""
    time_idx = header.index("time_since_boot_ms") if "time_since_boot_ms" in header else None
    if time_idx is None:
        return False
    for r in rows[1:]:
        if time_idx < len(r) and r[time_idx]:
            if is_synthetic(r[time_idx]):
                return True
    return False


def fix_file_v2(path, last_time_ms, is_new_distance):
    """
    Process row by row. Only replace rows with synthetic time_since_boot_ms (1000000-1010000).
    Skip files with no synthetic data - just update last_time_ms from real values.
    - When we hit 1000001 and last_time_ms is not None (same distance, no reboot): use last_time_ms + offset.
    - When we hit a dip (synthetic < last): config restart, use last + offset.
    - First file of distance: keep 1000001 as start (we rebooted).
    - Use tx_interval between packets.
    """
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "payload" not in rows[0]:
        return last_time_ms, False
    header = rows[0]
    payload_idx = header.index("payload")
    time_idx = header.index("time_since_boot_ms") if "time_since_boot_ms" in header else None
    tx_idx = header.index("tx_interval_ms") if "tx_interval_ms" in header else None
    if time_idx is None:
        return last_time_ms, False

    if not file_has_synthetic(rows, header):
        cur = last_time_ms
        for i in range(1, len(rows)):
            r = rows[i]
            payload = str(r[payload_idx]).strip() if len(r) > payload_idx else ""
            if payload.startswith("CFG "):
                continue
            if payload == "PACKET_LOST":
                if cur is not None and tx_idx is not None and len(r) > tx_idx and r[tx_idx]:
                    try:
                        cur = cur + float(r[tx_idx])
                    except (ValueError, TypeError):
                        pass
                continue
            if time_idx < len(r) and r[time_idx]:
                try:
                    cur = float(r[time_idx])
                except (ValueError, TypeError):
                    pass
        return (int(cur) if cur is not None else None), False

    current_ms = last_time_ms
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
            if current_ms is not None:
                new_ms = int(current_ms + tx_ms)
                r[time_idx] = str(new_ms)
                current_ms = new_ms
                modified = True
            continue

        t_val = r[time_idx] if time_idx < len(r) else ""
        tx_ms = 1550.0
        if tx_idx is not None and len(r) > tx_idx and r[tx_idx]:
            try:
                tx_ms = float(r[tx_idx])
            except (ValueError, TypeError):
                pass

        if not is_synthetic(t_val):
            try:
                current_ms = float(t_val) if t_val else current_ms
            except (ValueError, TypeError):
                pass
            continue

        t_ms = float(t_val)
        if current_ms is None or is_new_distance:
            new_ms = 1000001 if current_ms is None else int(current_ms + COUPLE_SECONDS_MS)
            is_new_distance = False
        else:
            if t_ms < current_ms or int(t_ms) == 1000001:
                new_ms = int(current_ms + COUPLE_SECONDS_MS)
            else:
                new_ms = int(current_ms + tx_ms)

        r[time_idx] = str(new_ms)
        parts = payload.strip('"').split(",")
        if parts:
            parts[0] = str(new_ms)
            r[payload_idx] = ",".join(parts)
        current_ms = new_ms + tx_ms
        modified = True

    if modified:
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
    return (int(current_ms) if current_ms is not None else None), modified


def main():
    dist_folders = sorted(
        [d for d in os.listdir(RAW_ROOT) if os.path.isdir(os.path.join(RAW_ROOT, d)) and d.startswith("distance_")],
        key=lambda x: parse_distance(x) or 0,
    )

    last_time_ms = None
    prev_distance = None

    for dn in dist_folders:
        dist_path = os.path.join(RAW_ROOT, dn)
        distance = parse_distance(dn)
        if distance is None:
            continue
        is_new_distance = prev_distance is not None
        prev_distance = distance

        for sf, bw, tp in get_file_order():
            path = find_file_in_dist(dist_path, sf, bw, tp)
            if not path or not os.path.isfile(path):
                continue
            last_time_ms, changed = fix_file_v2(path, last_time_ms, is_new_distance)
            if changed:
                rel = os.path.relpath(path, RAW_ROOT)
                print(f"  Fixed: {rel}")
        last_time_ms = None

    print("Done.")


if __name__ == "__main__":
    main()
