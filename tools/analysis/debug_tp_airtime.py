"""
Debug: compute exact airtime values for config tp as the plot script does.
"""
import csv
import os
import re
from collections import defaultdict
import numpy as np

WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
DATA_ROOT = os.path.join(WORKSPACE, "raw_test_data")
AIRTIME_FILENAME = "airtime_by_sf_bw_payload.csv"

BW_VALUES = [62500, 125000, 250000, 500000]
SF_VALUES = [7, 8, 9, 10, 11, 12]
TP_VALUES = [2, 12, 22]
CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")


def load_airtime_table():
    path = os.path.join(DATA_ROOT, AIRTIME_FILENAME)
    if not os.path.isfile(path):
        path = os.path.join(WORKSPACE, AIRTIME_FILENAME)
    by_sf_bw = defaultdict(list)
    with open(path, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                bw = int(row["bw"])
                sf = int(row["sf"])
                airtime = float(row["airtime_ms"])
                by_sf_bw[(sf, bw)].append(airtime)
            except (KeyError, ValueError, TypeError):
                continue
    return {k: np.mean(v) for k, v in by_sf_bw.items()}


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def payload_is_valid(payload):
    if payload == "PACKET_LOST":
        return False
    parts = (payload or "").strip('"').split(",")
    for part in parts:
        if part and not re.match(r"^[0-9A-F]+$", part):
            return False
    return True


def file_metrics(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "payload" not in rows[0]:
        return None
    header = rows[0]
    payload_idx = header.index("payload")
    total = 0
    lost = 0
    for r in rows[1:]:
        if len(r) <= payload_idx:
            continue
        if str(r[payload_idx]).strip().startswith("CFG "):
            continue
        total += 1
        if not payload_is_valid(r[payload_idx]):
            lost += 1
    if total == 0:
        return None
    return lost / total


def main():
    airtime_table = load_airtime_table()
    print(f"Loaded {len(airtime_table)} (sf,bw) airtimes from {os.path.join(DATA_ROOT, AIRTIME_FILENAME)}")
    print()

    # Simulate collect_by_config for dims=["tp"]
    by_cfg = defaultdict(list)  # key = (tp,)
    for dn in sorted(os.listdir(DATA_ROOT)):
        dpath = os.path.join(DATA_ROOT, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        distance = parse_distance(dn)
        if distance is None:
            continue
        for root, _, files in os.walk(dpath):
            for fn in files:
                m = CFG_RE.match(fn)
                if not m:
                    continue
                sf, bw, tp = map(int, m.groups())
                if bw not in BW_VALUES or tp not in TP_VALUES or sf not in SF_VALUES:
                    continue
                path = os.path.join(root, fn)
                per = file_metrics(path)
                if per is None:
                    continue
                airtime = airtime_table.get((sf, bw))
                key = (tp,)
                by_cfg[key].append((distance, per, airtime))

    # Aggregate per (tp, distance)
    result = {}
    for key, pts in by_cfg.items():
        by_dist = defaultdict(list)
        for d, per, airtime in pts:
            by_dist[d].append((per, airtime))
        out = []
        for d in sorted(by_dist.keys()):
            vals = by_dist[d]
            pers = [v[0] for v in vals]
            airtimes = [v[1] for v in vals if v[1] is not None]
            out.append((
                np.mean(pers),
                d,
                np.mean(airtimes) if airtimes else None,
            ))
        result[key] = out

    # Print points with PER > 0 (what gets plotted)
    print("Points with PER > 0 (plotted on PER vs airtime for config tp):")
    print("-" * 55)
    for tp in TP_VALUES:
        pts = result.get((tp,), [])
        for per, dist, airtime in pts:
            if per > 0 and airtime is not None:
                print(f"  TP{tp} distance={dist}m: PER={per*100:.2f}%, airtime={airtime:.2f} ms")

    # Unique airtime values
    all_airtimes = []
    for key, pts in result.items():
        for per, dist, airtime in pts:
            if per > 0 and airtime is not None:
                all_airtimes.append(airtime)

    if all_airtimes:
        print()
        print(f"Unique airtimes in plotted points: {sorted(set(round(a, 2) for a in all_airtimes))}")
        print(f"Min: {min(all_airtimes):.2f}, Max: {max(all_airtimes):.2f}, Mean: {np.mean(all_airtimes):.2f}")


if __name__ == "__main__":
    main()
