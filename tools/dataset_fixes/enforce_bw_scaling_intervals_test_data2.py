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


def parse_distance(folder):
    m = re.match(r"^distance_([\d.]+)m?$", folder)
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


def interval_values(rows):
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


def mean_interval(rows):
    vals = interval_values(rows)
    return statistics.mean(vals) if vals else None


def normalize_intervals(rows, target_mean):
    vals = interval_values(rows)
    if not vals:
        return False
    cur_mean = statistics.mean(vals)
    if cur_mean <= 0:
        return False

    scale = target_mean / cur_mean
    changed = False

    base_ts = parse_ts(rows[1][0] if len(rows) > 1 and len(rows[1]) > 0 else "")
    if base_ts is None:
        base_ts = parse_ts(rows[2][0] if len(rows) > 2 and len(rows[2]) > 0 else "")
    if base_ts is None:
        base_ts = datetime.now()

    cur_ts = base_ts
    for i in range(2, len(rows)):
        r = rows[i]
        if len(r) < 5:
            continue
        if i == 2:
            # First packet interval can be misleading; pin to target mean.
            new_t = int(max(100, round(target_mean)))
        else:
            try:
                t = float(r[4]) if r[4] else target_mean
            except Exception:
                t = target_mean
            new_t = int(max(100, round(t * scale)))

        if r[4] != str(new_t):
            r[4] = str(new_t)
            changed = True

        cur_ts = cur_ts + timedelta(milliseconds=new_t)
        if len(r) > 0:
            new_ts = cur_ts.isoformat()
            if r[0] != new_ts:
                r[0] = new_ts
                changed = True
    return changed


def main():
    # Group by (distance, sf, tp), then enforce BW inverse scaling.
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
            mt = mean_interval(rows)
            if mt is None:
                continue
            payload = rows[1][5] if len(rows[1]) > 5 else ""
            groups.setdefault((dist, sf, tp), []).append(
                {
                    "bw": bw,
                    "path": fp,
                    "rows": rows,
                    "mean": mt,
                    "filled": FILLED_MARK in payload,
                    "generated": GENERATED_MARK in payload,
                }
            )

    updated = 0
    for _, items in groups.items():
        # Use unfilled & ungenerated anchors first.
        anchors = [x for x in items if (not x["filled"] and not x["generated"])]
        if not anchors:
            anchors = [x for x in items if not x["filled"]]
        if not anchors:
            anchors = items

        # time * bw should be approximately constant for same sf,tp,distance
        c_values = [a["mean"] * a["bw"] for a in anchors]
        c = statistics.mean(c_values)

        for it in items:
            if not it["filled"]:
                continue
            target = c / it["bw"]
            if normalize_intervals(it["rows"], target):
                write_rows(it["path"], it["rows"])
                updated += 1

    print(f"Updated filled files with BW scaling: {updated}")


if __name__ == "__main__":
    main()

