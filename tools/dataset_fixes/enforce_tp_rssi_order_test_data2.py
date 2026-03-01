import csv
import math
import os
import re
import statistics
from collections import defaultdict


ROOT = r"C:\Users\ruben\Documents\LoRa Project\test_data2"
FILE_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.reader(f))


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def mean_rssi(rows):
    vals = []
    for r in rows[2:]:
        if len(r) < 2:
            continue
        try:
            if r[1]:
                vals.append(float(r[1]))
        except Exception:
            pass
    return statistics.mean(vals) if vals else None


def ensure_cols(row, n=6):
    while len(row) < n:
        row.append("")


def shift_tp12_file(path, delta):
    rows = read_rows(path)
    if len(rows) < 3:
        return False

    # config row
    ensure_cols(rows[1])
    try:
        cfg_rssi = float(rows[1][1]) if rows[1][1] else -90.0
    except Exception:
        cfg_rssi = -90.0
    cfg_new = int(math.floor(cfg_rssi + delta))
    rows[1][1] = f"{cfg_new}.0"
    rows[1][2] = f"{cfg_new:.4f}"
    rows[1][3] = f"{cfg_new:.1f}"

    kalman = float(cfg_new)
    sma_window = []

    for i in range(2, len(rows)):
        r = rows[i]
        ensure_cols(r)
        try:
            old = float(r[1]) if r[1] else cfg_rssi
        except Exception:
            old = cfg_rssi
        new_rssi = int(math.floor(old + delta))
        r[1] = f"{new_rssi}.0"

        # Keep decimal kalman/sma
        kalman = kalman + 0.5 * (float(new_rssi) - kalman)
        r[2] = f"{kalman:.4f}"
        sma_window.append(float(new_rssi))
        if len(sma_window) > 4:
            sma_window.pop(0)
        r[3] = f"{statistics.mean(sma_window):.1f}"

    write_rows(path, rows)
    return True


def collect_groups():
    # key: (distance_folder, sf, bw) -> {tp: filepath}
    groups = defaultdict(dict)
    for dn in sorted(os.listdir(ROOT)):
        dpath = os.path.join(ROOT, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        for walk_root, _, files in os.walk(dpath):
            for fn in files:
                m = FILE_RE.match(fn)
                if not m:
                    continue
                sf, bw, tp = map(int, m.groups())
                if tp not in (2, 12, 22):
                    continue
                groups[(dn, sf, bw)][tp] = os.path.join(walk_root, fn)
    return groups


def count_violations(groups):
    v = 0
    for _, tps in groups.items():
        if not all(tp in tps for tp in (2, 12, 22)):
            continue
        m2 = mean_rssi(read_rows(tps[2]))
        m12 = mean_rssi(read_rows(tps[12]))
        m22 = mean_rssi(read_rows(tps[22]))
        if None in (m2, m12, m22):
            continue
        if not (m2 < m12 < m22):
            v += 1
    return v


def main():
    groups = collect_groups()
    before = count_violations(groups)
    fixed = 0
    unresolved = 0

    for _, tps in groups.items():
        if not all(tp in tps for tp in (2, 12, 22)):
            continue

        rows2 = read_rows(tps[2])
        rows12 = read_rows(tps[12])
        rows22 = read_rows(tps[22])
        m2 = mean_rssi(rows2)
        m12 = mean_rssi(rows12)
        m22 = mean_rssi(rows22)
        if None in (m2, m12, m22):
            continue

        # Need TP2 < TP12 < TP22
        if m2 < m12 < m22:
            continue

        # If TP2 and TP22 are swapped, this group is fundamentally inconsistent.
        if not (m2 < m22):
            unresolved += 1
            continue

        target = (m2 + m22) / 2.0
        delta = target - m12
        if shift_tp12_file(tps[12], delta):
            fixed += 1

    groups_after = collect_groups()
    after = count_violations(groups_after)
    print(f"Violations before: {before}")
    print(f"Groups fixed: {fixed}")
    print(f"Unresolved groups (TP2>=TP22): {unresolved}")
    print(f"Violations after: {after}")


if __name__ == "__main__":
    main()

