"""
Analyze: for config tp, what airtime do PER>0 points have, and which (SF,BW) configs does it map to?
"""
import csv
import os
import re
from collections import defaultdict

WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
DATA_ROOT = os.path.join(WORKSPACE, "raw_test_data")
AIRTIME_FILE = os.path.join(WORKSPACE, "airtime_by_sf_bw_payload.csv")

BW_VALUES = [62500, 125000, 250000, 500000]
SF_VALUES = [7, 8, 9, 10, 11, 12]
TP_VALUES = [2, 12, 22]
CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")


def load_airtime_table():
    by_sf_bw = defaultdict(list)
    with open(AIRTIME_FILE, "r", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                bw = int(row["bw"])
                sf = int(row["sf"])
                airtime = float(row["airtime_ms"])
                by_sf_bw[(sf, bw)].append(airtime)
            except (KeyError, ValueError, TypeError):
                continue
    return {k: sum(v) / len(v) for k, v in by_sf_bw.items()}


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


def file_per_and_airtime(path, sf, bw, airtime_table):
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "payload" not in rows[0]:
        return None, None
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
        return None, None
    per = lost / total
    airtime = airtime_table.get((sf, bw))
    return per, airtime


def main():
    airtime_table = load_airtime_table()

    # 1. Find which (SF,BW) configs actually exist in raw_test_data
    existing_sfbw = set()
    for dn in sorted(os.listdir(DATA_ROOT)):
        dpath = os.path.join(DATA_ROOT, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        for root, _, files in os.walk(dpath):
            for fn in files:
                m = CFG_RE.match(fn)
                if m:
                    sf, bw, _ = map(int, m.groups())
                    if bw in BW_VALUES and sf in SF_VALUES:
                        existing_sfbw.add((sf, bw))

    # 2. Mean airtime across configs that EXIST in the data (not all 24)
    existing_airtimes = [airtime_table[(sf, bw)] for (sf, bw) in existing_sfbw if (sf, bw) in airtime_table]
    mean_airtime = sum(existing_airtimes) / len(existing_airtimes) if existing_airtimes else 0
    print(f"Airtime where all config-tp points appear: {mean_airtime:.2f} ms")
    print(f"  (Mean across {len(existing_sfbw)} (SF,BW) configs present in raw_test_data)")
    print()
    print("Configs that contribute to this airtime:")
    for (sf, bw) in sorted(existing_sfbw):
        at = airtime_table.get((sf, bw))
        bw_str = f"{bw/1000:.1f}" if bw % 1000 else str(bw//1000)
        print(f"  SF{sf} BW{bw_str} kHz: {at:.1f} ms" if at else f"  SF{sf} BW{bw}: N/A")
    print()

    # 3. Per (SF,BW): airtime and which have packet loss
    per_by_sfbw = defaultdict(list)  # (sf,bw) -> [per from each file]
    for dn in sorted(os.listdir(DATA_ROOT)):
        dpath = os.path.join(DATA_ROOT, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        for root, _, files in os.walk(dpath):
            for fn in files:
                m = CFG_RE.match(fn)
                if not m:
                    continue
                sf, bw, tp = map(int, m.groups())
                if bw not in BW_VALUES or sf not in SF_VALUES:
                    continue
                path = os.path.join(root, fn)
                per, airtime = file_per_and_airtime(path, sf, bw, airtime_table)
                if per is not None:
                    per_by_sfbw[(sf, bw)].append((per, airtime))

    # Average PER per (SF,BW)
    sfbw_with_loss = []
    for (sf, bw), vals in per_by_sfbw.items():
        avg_per = sum(v[0] for v in vals) / len(vals)
        airtime = vals[0][1] if vals else None  # same for all distances
        if avg_per > 0:
            sfbw_with_loss.append((sf, bw, avg_per * 100, airtime))

    print("Configs (SF,BW) with packet loss (PER > 0):")
    print("-" * 50)
    for sf, bw, per_pct, airtime in sorted(sfbw_with_loss, key=lambda x: -x[2]):
        bw_khz = bw / 1000 if bw % 1000 else bw // 1000
        print(f"  SF{sf} BW{bw} ({bw_khz} kHz): PER={per_pct:.2f}%, airtime={airtime:.1f} ms" if airtime else f"  SF{sf} BW{bw}: PER={per_pct:.2f}%, airtime=N/A")

    print()
    print("Airtimes of configs with packet loss:")
    airtimes_with_loss = sorted(set(a for _, _, _, a in sfbw_with_loss if a is not None))
    print(f"  {airtimes_with_loss}")

    # 4. Which airtime is most common among lossy configs?
    if sfbw_with_loss:
        from collections import Counter
        airtime_counts = Counter(a for _, _, _, a in sfbw_with_loss if a is not None)
        if airtime_counts:
            most_common = airtime_counts.most_common(1)[0]
            print()
            print(f"Most common airtime among configs with loss: {most_common[0]:.1f} ms ({most_common[1]} configs)")


if __name__ == "__main__":
    main()
