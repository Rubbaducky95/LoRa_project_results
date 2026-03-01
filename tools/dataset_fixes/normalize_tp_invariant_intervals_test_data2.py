import csv
import os
import re
import statistics
from datetime import datetime, timedelta


ROOT = r"C:\Users\ruben\Documents\LoRa Project\test_data2"
FILLED_MARK = "[FILLED_ROWS]"
GENERATED_MARK = "[GENERATED_FROM_RAW]"


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def parse_ts(text):
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.reader(f))


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def time_values(rows):
    vals = []
    for r in rows[2:]:
        if len(r) < 5:
            continue
        try:
            if r[4]:
                vals.append(float(r[4]))
        except Exception:
            pass
    return vals


def mean_time(rows):
    vals = time_values(rows)
    return statistics.mean(vals) if vals else None


def normalize_file_times(rows, target_mean):
    vals = time_values(rows)
    if not vals:
        return False
    current_mean = statistics.mean(vals)
    if current_mean <= 0:
        return False

    scale = target_mean / current_mean
    changed = False

    # Keep timestamps consistent with adjusted intervals.
    base_ts = parse_ts(rows[1][0] if len(rows) > 1 and len(rows[1]) > 0 else "")
    if base_ts is None:
        base_ts = parse_ts(rows[2][0] if len(rows) > 2 and len(rows[2]) > 0 else "")
    if base_ts is None:
        base_ts = datetime.now()
    current_ts = base_ts

    for i in range(2, len(rows)):
        r = rows[i]
        if len(r) < 5:
            continue
        try:
            t = float(r[4]) if r[4] else target_mean
        except Exception:
            t = target_mean
        new_t = int(max(100, round(t * scale)))
        if r[4] != str(new_t):
            r[4] = str(new_t)
            changed = True
        current_ts = current_ts + timedelta(milliseconds=new_t)
        if len(r) > 0:
            new_ts = current_ts.isoformat()
            if r[0] != new_ts:
                r[0] = new_ts
                changed = True
    return changed


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
            rows = read_rows(fp)
            if len(rows) < 3:
                continue
            payload = rows[1][5] if len(rows[1]) > 5 else ""
            mt = mean_time(rows)
            if mt is None:
                continue

            key = (dist, sf, bw)
            groups.setdefault(key, []).append(
                {
                    "path": fp,
                    "rows": rows,
                    "tp": tp,
                    "mean_time": mt,
                    "filled": FILLED_MARK in payload,
                    "generated": GENERATED_MARK in payload,
                }
            )

    updated = 0
    for _, items in groups.items():
        # Prefer unfilled + ungenerated files as reference.
        anchor = [x["mean_time"] for x in items if (not x["filled"] and not x["generated"])]
        if not anchor:
            # Then use non-filled if no pure raw clone exists.
            anchor = [x["mean_time"] for x in items if not x["filled"]]
        if not anchor:
            anchor = [x["mean_time"] for x in items]

        target_mean = statistics.mean(anchor)

        for it in items:
            if not it["filled"]:
                continue
            if normalize_file_times(it["rows"], target_mean):
                write_rows(it["path"], it["rows"])
                updated += 1

    print(f"Updated filled files: {updated}")


if __name__ == "__main__":
    main()

