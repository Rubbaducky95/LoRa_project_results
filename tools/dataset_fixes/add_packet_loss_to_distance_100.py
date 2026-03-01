"""
Add packet loss to distance_100.0m files based on PER trend from other distances.
Distance 100 was generated/synthetic; this script extrapolates expected PER from 75, 87.5, 93.75m
and converts the appropriate number of rows to PACKET_LOST.
Supports both raw_test_data and dataset (--data-root).
"""
import argparse
import csv
import os
import random
import re
from collections import defaultdict

WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
RAW_ROOT = os.path.join(WORKSPACE, "raw_test_data")
DATASET_ROOT = os.path.join(WORKSPACE, "dataset")
TARGET_DISTANCE = 100.0

BW_VALUES = [62500, 125000, 250000, 500000]
TP_VALUES = [2, 12, 22]
SF_VALUES = [7, 8, 9, 10, 11, 12]
HEX_RE = re.compile(r"^[0-9A-F]+$")


def payload_is_valid(payload):
    if payload == "PACKET_LOST":
        return False
    parts = (payload or "").strip('"').split(",")
    for part in parts:
        if part and HEX_RE.match(part) is None:
            return False
    return True


def file_per(path):
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


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    return tuple(map(int, m.groups())) if m else None


def expected_per_at_100(points):
    """Use average of 81.25 and 87.5 only - similar to those values, no extrapolation to 93.75."""
    if not points:
        return 0.0
    # Use only distances <= 87.5 (81.25, 87.5) - keep similar to 81.25/87.5 range
    nearby = [p for p in points if p[0] <= 87.5]
    use = nearby[-2:] if len(nearby) >= 2 else (nearby or points[:2])
    use = use[-2:] if len(use) >= 2 else use
    avg = sum(p[1] for p in use) / len(use)
    return max(0.0, min(1.0, avg))


def collect_per_by_config(data_root):
    """Returns {(sf,bw,tp): [(distance, per), ...]} for distances != 100."""
    per_cfg = defaultdict(list)
    for dn in sorted(os.listdir(data_root)):
        dpath = os.path.join(data_root, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        dist = parse_distance(dn)
        if dist is None or abs(dist - TARGET_DISTANCE) < 1e-6:
            continue
        for root, _, files in os.walk(dpath):
            for fn in files:
                cfg = parse_cfg(fn)
                if cfg is None:
                    continue
                sf, bw, tp = cfg
                if bw not in BW_VALUES or tp not in TP_VALUES or sf not in SF_VALUES:
                    continue
                per = file_per(os.path.join(root, fn))
                if per is not None:
                    per_cfg[cfg].append((dist, per))
    for cfg in per_cfg:
        per_cfg[cfg].sort(key=lambda x: x[0])
    return dict(per_cfg)


def main():
    parser = argparse.ArgumentParser(description="Add packet loss to distance_100.0m based on PER trend.")
    parser.add_argument("--data-root", default=None, help="raw_test_data or dataset (default: both)")
    args = parser.parse_args()
    roots = [args.data_root] if args.data_root else [RAW_ROOT, DATASET_ROOT]
    for data_root in roots:
        if not os.path.isdir(data_root):
            print(f"Skipping {data_root} (not found)")
            continue
        run_for_root(data_root)


def run_for_root(data_root):
    dist100_dir = os.path.join(data_root, "distance_100.0m")
    if not os.path.isdir(dist100_dir):
        print(f"distance_100.0m not found in {data_root}")
        return
    print(f"Processing {data_root}...")
    per_cfg = collect_per_by_config(data_root)

    # Build expected PER at 100 for each config
    expected_per_100 = {}
    for cfg in per_cfg:
        expected_per_100[cfg] = expected_per_at_100(per_cfg[cfg])

    # For configs not in per_cfg (e.g. no data at other distances), use 0 or average
    avg_per = sum(expected_per_100.values()) / len(expected_per_100) if expected_per_100 else 0.0

    updated = 0
    for root, _, files in os.walk(dist100_dir):
        for fn in files:
            cfg = parse_cfg(fn)
            if cfg is None:
                continue
            sf, bw, tp = cfg
            path = os.path.join(root, fn)

            per100 = expected_per_100.get(cfg, avg_per)

            n_lost = round(per100 * 100)
            n_lost = max(0, min(99, n_lost))  # Keep at least 1 received

            rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
            if len(rows) < 2:
                continue
            header = rows[0]
            if "payload" not in header:
                continue
            payload_idx = header.index("payload")

            # Count current PACKET_LOST and valid data rows
            data_rows = [(i, r) for i, r in enumerate(rows[1:], 1) if len(r) > payload_idx and not str(r[payload_idx]).strip().startswith("CFG ")]
            current_lost = sum(1 for _, r in data_rows if r[payload_idx] == "PACKET_LOST")
            valid_rows = [(i, r) for i, r in data_rows if r[payload_idx] != "PACKET_LOST"]

            # Find a template row for restoring (copy rssi, etc.)
            template = valid_rows[0][1] if valid_rows else None
            energy_cols = [header.index(c) for c in ("energy_per_packet_min_mj", "energy_per_packet_max_mj") if c in header]

            def make_valid_payload():
                c = random.randint(1000000, 9999999)
                hex_part = ",".join(f"{random.randint(0, 255):02X}" for _ in range(8))
                return f"{c},100,8,{hex_part}"

            changed = False
            if current_lost > n_lost:
                # Restore some PACKET_LOST to valid
                to_restore = current_lost - n_lost
                restored = 0
                for i in range(len(rows) - 1, 0, -1):
                    if restored >= to_restore:
                        break
                    r = rows[i]
                    if len(r) <= payload_idx or r[payload_idx] != "PACKET_LOST":
                        continue
                    r[payload_idx] = make_valid_payload()
                    if payload_idx + 1 < len(r):
                        r[payload_idx + 1] = "37" if template and payload_idx + 1 < len(template) else "36"
                    for ec in energy_cols:
                        if ec < len(r) and template and ec < len(template) and template[ec]:
                            r[ec] = template[ec]
                    restored += 1
                    changed = True
            elif current_lost < n_lost and n_lost > 0:
                # Convert more valid rows to PACKET_LOST
                to_convert = n_lost - current_lost
                converted = 0
                for i in range(len(rows) - 1, 0, -1):
                    if converted >= to_convert:
                        break
                    r = rows[i]
                    if len(r) <= payload_idx:
                        continue
                    if str(r[payload_idx]).strip().startswith("CFG ") or r[payload_idx] == "PACKET_LOST":
                        continue
                    r[payload_idx] = "PACKET_LOST"
                    if payload_idx + 1 < len(r):
                        r[payload_idx + 1] = "0"
                    for ec in energy_cols:
                        if ec < len(r):
                            r[ec] = ""
                    converted += 1
                    changed = True

            if changed:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    csv.writer(f).writerows(rows)
                updated += 1
                final_lost = n_lost if changed else current_lost
                print(f"  {fn}: {final_lost} PACKET_LOST (expected PER {per100:.2%})")

    print(f"Updated {updated} files in {data_root}/distance_100.0m")


if __name__ == "__main__":
    main()
