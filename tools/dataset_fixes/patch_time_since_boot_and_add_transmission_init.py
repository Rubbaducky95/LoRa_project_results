"""
Patch time_since_boot_ms and add time_since_transmission_init_ms.
- Test RESTARTS at each distance: reset both timers when changing distance.
- First packet of each distance (SF7 BW62500 TP2): time_since_boot_ms = first value from payload,
  time_since_transmission_init_ms = 0.
- Subsequent packets: both increase by prev tx_interval_ms.
- For PACKET_LOST (no payload): use prev + prev_tx_interval for both.
"""
import csv
import os
import re

WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
DATA_ROOT = os.path.join(WORKSPACE, "raw_test_data")

BW_VALUES = [62500, 125000, 250000, 500000]
TP_VALUES = [2, 12, 22]
SF_VALUES = [7, 8, 9, 10, 11, 12]
DEFAULT_TX_INTERVAL = 1550.0
FIRST_CONFIG = (7, 62500, 2)  # SF7, BW62500, TP2


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    return tuple(map(int, m.groups())) if m else None


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def get_file_order():
    for sf in SF_VALUES:
        for bw in BW_VALUES:
            for tp in TP_VALUES:
                yield (sf, bw, tp)


def find_file_in_dist(dist_path, sf, bw, tp):
    for sub in os.listdir(dist_path):
        spath = os.path.join(dist_path, sub)
        if not os.path.isdir(spath):
            continue
        for fn in os.listdir(spath):
            cfg = parse_cfg(fn)
            if cfg and cfg == (sf, bw, tp):
                return os.path.join(spath, fn)
    return None


def parse_float(s):
    try:
        return float(s.strip()) if s else None
    except (ValueError, TypeError):
        return None


def payload_first_value_ms(payload):
    """Extract first value from payload as ms since boot (sender timestamp). Returns None if invalid."""
    if not payload or payload == "PACKET_LOST" or str(payload).strip().startswith("CFG "):
        return None
    parts = str(payload).strip('"').split(",")
    if not parts or not parts[0].strip():
        return None
    try:
        return float(parts[0].strip())
    except (ValueError, TypeError):
        return None


def update_file(path, state, is_first_packet_of_distance):
    """
    Process one CSV file. state = {last_time_boot, last_time_transmission_init, last_tx_interval}.
    is_first_packet_of_distance: True for first packet of SF7 BW62500 TP2 in this distance.
    Returns updated state and True if file was modified.
    """
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return state, False
    header = rows[0]
    if "time_since_boot_ms" not in header or "tx_interval_ms" not in header:
        return state, False

    time_idx = header.index("time_since_boot_ms")
    tx_idx = header.index("tx_interval_ms")
    payload_idx = header.index("payload") if "payload" in header else -1

    has_transmission_init = "time_since_transmission_init_ms" in header
    if has_transmission_init:
        init_idx = header.index("time_since_transmission_init_ms")
    else:
        insert_idx = time_idx + 1
        header.insert(insert_idx, "time_since_transmission_init_ms")
        init_idx = insert_idx

    modified = False
    first_packet_seen = is_first_packet_of_distance

    for i in range(1, len(rows)):
        row = rows[i]
        while len(row) < max(time_idx, tx_idx, init_idx) + 1:
            row.append("")

        is_cfg = payload_idx >= 0 and len(row) > payload_idx and str(row[payload_idx]).strip().startswith("CFG ")
        if is_cfg:
            if not has_transmission_init:
                row.insert(insert_idx, "")
                modified = True
            continue

        payload = row[payload_idx] if payload_idx >= 0 and len(row) > payload_idx else ""
        tx_ms = parse_float(row[tx_idx]) if tx_idx < len(row) else None
        tx_ms = tx_ms or DEFAULT_TX_INTERVAL

        # time_since_boot_ms: from payload first value when available, else prev + prev_tx_interval
        payload_first = payload_first_value_ms(payload)
        if payload_first is not None:
            time_ms = payload_first
        else:
            time_ms = state["last_time_boot"] + (state["last_tx_interval"] or DEFAULT_TX_INTERVAL) if state["last_time_boot"] is not None else 0

        # time_since_transmission_init_ms: 0 for first packet of distance, else prev + prev_tx_interval
        if first_packet_seen:
            time_since_init = 0
            first_packet_seen = False
        else:
            time_since_init = state["last_time_transmission_init"] + (state["last_tx_interval"] or DEFAULT_TX_INTERVAL)

        row[time_idx] = f"{time_ms:.0f}"
        if has_transmission_init:
            row[init_idx] = f"{time_since_init:.0f}"
        else:
            row.insert(insert_idx, f"{time_since_init:.0f}")
        modified = True

        state["last_time_boot"] = time_ms
        state["last_time_transmission_init"] = time_since_init
        state["last_tx_interval"] = tx_ms

    if modified:
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
    return state, modified


def main():
    dist_folders = sorted(
        [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith("distance_")],
        key=lambda x: parse_distance(x) or 0,
    )
    total_modified = 0

    for dn in dist_folders:
        dpath = os.path.join(DATA_ROOT, dn)
        if parse_distance(dn) is None:
            continue
        state = {"last_time_boot": None, "last_time_transmission_init": 0, "last_tx_interval": None}
        first_config = True

        for sf, bw, tp in get_file_order():
            path = find_file_in_dist(dpath, sf, bw, tp)
            if not path or not os.path.isfile(path):
                continue
            is_first = first_config and (sf, bw, tp) == FIRST_CONFIG
            state, changed = update_file(path, state, is_first_packet_of_distance=is_first)
            if changed:
                total_modified += 1
                print(f"  Patched: {os.path.relpath(path, DATA_ROOT)}")
            first_config = False

    print(f"\nPatched {total_modified} files. Reset at each distance; time_since_boot from payload at start.")


if __name__ == "__main__":
    main()
