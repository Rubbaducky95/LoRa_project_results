import csv
import os
import re
import statistics
from collections import deque


ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
CSV_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")
WINDOW = 4


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None


def ensure_col(header, name):
    if name not in header:
        header.append(name)
    return header.index(name)


def process_file(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8")))
    if len(rows) < 3:
        return False

    header = rows[0]
    if "rssi" not in header:
        return False
    rssi_idx = header.index("rssi")
    sma_idx = ensure_col(header, "sma_rssi_recalc")

    # Normalize row lengths
    for i in range(1, len(rows)):
        while len(rows[i]) < len(header):
            rows[i].append("")

    # x0 seed = average RSSI in file
    rssi_vals = []
    for r in rows[2:]:
        z = parse_float(r[rssi_idx]) if len(r) > rssi_idx else None
        if z is not None:
            rssi_vals.append(z)
    if not rssi_vals:
        return False
    x0 = sum(rssi_vals) / len(rssi_vals)

    # Config row holds seed value for traceability
    rows[1][sma_idx] = f"{x0:.1f}"

    # Seed deque with x0, then roll through packet RSSI values.
    buf = deque([x0], maxlen=WINDOW)
    for i in range(2, len(rows)):
        r = rows[i]
        z = parse_float(r[rssi_idx]) if len(r) > rssi_idx else None
        if z is None:
            r[sma_idx] = ""
            continue
        buf.append(z)
        r[sma_idx] = f"{statistics.mean(buf):.1f}"

    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    return True


def main():
    updated = 0
    for root, _, files in os.walk(ROOT):
        for fn in files:
            if CSV_RE.match(fn) is None:
                continue
            if process_file(os.path.join(root, fn)):
                updated += 1
    print(f"Updated files with sma_rssi_recalc: {updated}")
    print(f"SMA window: {WINDOW}, seed x0 = file-average RSSI")


if __name__ == "__main__":
    main()

