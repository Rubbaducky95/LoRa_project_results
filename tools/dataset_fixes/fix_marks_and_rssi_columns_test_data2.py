import csv
import math
import os
import re
import statistics


TEST_ROOT = r"C:\Users\ruben\Documents\LoRa Project\test_data2"
RAW_ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"

MARK_GENERATED = "[GENERATED_FROM_RAW]"
MARK_FILLED = "[FILLED_ROWS]"


def ensure_len(row, n):
    while len(row) < n:
        row.append("")


def parse_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def file_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.reader(f))


def is_distance_dir(name):
    return name.startswith("distance_")


def fix_file(test_path, raw_path):
    rows = file_rows(test_path)
    if len(rows) < 2:
        return False

    raw_rows = None
    if os.path.exists(raw_path):
        try:
            raw_rows = file_rows(raw_path)
        except Exception:
            raw_rows = None

    # Mark files that differ from raw (or do not exist in raw) and are not already generated.
    differs_from_raw = (raw_rows is None) or (rows != raw_rows)

    ensure_len(rows[1], 6)
    cfg_payload = rows[1][5] or ""
    has_generated_mark = MARK_GENERATED in cfg_payload
    has_filled_mark = MARK_FILLED in cfg_payload

    # Apply this pass to generated files and filled/modified files.
    should_fix = has_generated_mark or differs_from_raw
    changed = False

    if should_fix and (not has_generated_mark) and (not has_filled_mark):
        rows[1][5] = (cfg_payload + " " + MARK_FILLED).strip()
        changed = True

    # Only floor "rssi" and keep kalman/sma as decimal computations.
    if should_fix:
        cfg_rssi = parse_float(rows[1][1], None)
        if cfg_rssi is not None:
            cfg_rssi_floor = int(math.floor(cfg_rssi))
            new_cfg_rssi = f"{cfg_rssi_floor}.0"
            if rows[1][1] != new_cfg_rssi:
                rows[1][1] = new_cfg_rssi
                changed = True

        # Recompute kalman/sma from RSSI series with decimal outputs.
        # Keep RSSI as floored doubles.
        k_val = 0.5
        kalman = parse_float(rows[1][2], None)
        if kalman is None:
            kalman = parse_float(rows[1][1], -90.0)
        window = []

        for i in range(2, len(rows)):
            ensure_len(rows[i], 6)
            row = rows[i]

            rssi_val = parse_float(row[1], None)
            if rssi_val is None:
                rssi_val = parse_float(rows[1][1], -90.0)
            rssi_floor = int(math.floor(rssi_val))
            rssi_float = float(rssi_floor)
            rssi_text = f"{rssi_floor}.0"
            if row[1] != rssi_text:
                row[1] = rssi_text
                changed = True

            kalman = kalman + k_val * (rssi_float - kalman)
            kalman_text = f"{kalman:.4f}"
            if row[2] != kalman_text:
                row[2] = kalman_text
                changed = True

            window.append(rssi_float)
            if len(window) > 4:
                window.pop(0)
            sma = statistics.mean(window) if window else rssi_float
            sma_text = f"{sma:.1f}"
            if row[3] != sma_text:
                row[3] = sma_text
                changed = True

    if changed:
        with open(test_path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
    return changed


def main():
    changed_files = 0
    for dn in os.listdir(TEST_ROOT):
        test_dir = os.path.join(TEST_ROOT, dn)
        if not (os.path.isdir(test_dir) and is_distance_dir(dn)):
            continue
        raw_dir = os.path.join(RAW_ROOT, dn)
        for fn in os.listdir(test_dir):
            if not fn.endswith(".csv"):
                continue
            test_path = os.path.join(test_dir, fn)
            raw_path = os.path.join(raw_dir, fn)
            if fix_file(test_path, raw_path):
                changed_files += 1

    print(f"Updated files: {changed_files}")


if __name__ == "__main__":
    main()

