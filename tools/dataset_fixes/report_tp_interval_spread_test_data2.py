import csv
import os
import re
import statistics


ROOT = r"C:\Users\ruben\Documents\LoRa Project\test_data2"
FILLED_MARK = "[FILLED_ROWS]"


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def mean_time(rows):
    vals = []
    for r in rows[2:]:
        if len(r) < 5:
            continue
        try:
            if r[4]:
                vals.append(float(r[4]))
        except Exception:
            pass
    return statistics.mean(vals) if vals else None


def main():
    groups = {}
    for dn in os.listdir(ROOT):
        folder = os.path.join(ROOT, dn)
        if not (os.path.isdir(folder) and dn.startswith("distance_")):
            continue
        dist = parse_distance(dn)
        if dist is None:
            continue
        for fn in os.listdir(folder):
            cfg = parse_cfg(fn)
            if not cfg:
                continue
            sf, bw, tp = cfg
            fp = os.path.join(folder, fn)
            rows = list(csv.reader(open(fp, "r", encoding="utf-8")))
            mt = mean_time(rows)
            if mt is None:
                continue
            payload = rows[1][5] if len(rows) > 1 and len(rows[1]) > 5 else ""
            filled = FILLED_MARK in payload
            key = (dist, sf, bw)
            groups.setdefault(key, []).append((tp, mt, filled, fp))

    spreads = []
    filled_spreads = []
    for key, items in groups.items():
        if len(items) < 2:
            continue
        means = [x[1] for x in items]
        spread = max(means) - min(means)
        spreads.append(spread)

        filled_means = [x[1] for x in items if x[2]]
        if len(filled_means) >= 2:
            filled_spreads.append(max(filled_means) - min(filled_means))

    print(f"group_count={len(spreads)}")
    print(f"overall_max_spread_ms={max(spreads) if spreads else 0:.2f}")
    print(f"overall_median_spread_ms={statistics.median(spreads) if spreads else 0:.2f}")
    print(f"filled_group_count={len(filled_spreads)}")
    print(f"filled_max_spread_ms={max(filled_spreads) if filled_spreads else 0:.2f}")
    print(f"filled_median_spread_ms={statistics.median(filled_spreads) if filled_spreads else 0:.2f}")


if __name__ == "__main__":
    main()

