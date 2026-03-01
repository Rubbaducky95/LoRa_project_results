"""
Count how many times time_since_boot_ms resets (goes backwards) within each distance folder.
Expected: 0 resets per distance (only when changing distance). Extra resets may indicate issues.
"""
import os
import re
import sys

import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(WORKSPACE, "raw_test_data")

BW_VALUES = [62500, 125000, 250000, 500000]
TP_VALUES = [2, 12, 22]
SF_VALUES = [7, 8, 9, 10, 11, 12]


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


def yield_times_from_file(path):
    """Yield (time_since_boot_ms, packet_idx_in_file) for each packet."""
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "time_since_boot_ms" not in rows[0]:
        return
    header = rows[0]
    time_idx = header.index("time_since_boot_ms")
    payload_idx = header.index("payload") if "payload" in header else -1
    pkt_idx = 0
    for r in rows[1:]:
        if payload_idx >= 0 and len(r) > payload_idx and str(r[payload_idx]).strip().startswith("CFG "):
            continue
        if len(r) <= time_idx or not r[time_idx]:
            continue
        try:
            yield float(r[time_idx]), pkt_idx
            pkt_idx += 1
        except (ValueError, TypeError):
            continue


def main():
    dist_folders = sorted(
        [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith("distance_")],
        key=lambda x: parse_distance(x) or 0,
    )
    details = []
    reset_locations = []

    for dn in dist_folders:
        dpath = os.path.join(DATA_ROOT, dn)
        distance = parse_distance(dn)
        if distance is None:
            continue
        resets = 0
        packets = 0
        prev_ms = None
        prev_cfg = None
        prev_global_idx = 0
        global_idx = 0

        for sf, bw, tp in get_file_order():
            path = find_file_in_dist(dpath, sf, bw, tp)
            if not path or not os.path.isfile(path):
                continue
            cfg = f"SF{sf}_BW{bw//1000}_TP{tp}"
            rel_path = os.path.relpath(path, DATA_ROOT)
            for t_ms, pkt_in_file in yield_times_from_file(path):
                packets += 1
                if prev_ms is not None and t_ms < prev_ms:
                    resets += 1
                    reset_locations.append({
                        "distance": dn,
                        "file": rel_path,
                        "config": cfg,
                        "packet_global_idx": global_idx,
                        "packet_in_file": pkt_in_file,
                        "prev_time_ms": prev_ms,
                        "curr_time_ms": t_ms,
                        "prev_config": prev_cfg,
                    })
                prev_ms = t_ms
                prev_cfg = cfg
                prev_global_idx = global_idx
                global_idx += 1

        details.append((dn, distance, resets, packets))

    print("Time resets (time_since_boot_ms goes backwards) within each distance folder:")
    print("-" * 65)
    for dn, dist, resets, packets in details:
        status = "OK" if resets == 0 else "*** RESETS ***"
        print(f"  {dn}: {resets} resets, {packets:,} packets  {status}")
    print("-" * 65)
    total = sum(d[2] for d in details)
    print(f"Total resets within distance folders: {total}\n")

    if reset_locations:
        print("Reset locations (where time went backwards):")
        print("-" * 65)
        for i, r in enumerate(reset_locations, 1):
            prev_cfg = r.get("prev_config", "?")
            print(f"  {i:3}. {r['distance']} | {r['config']} | packet {r['packet_in_file']} (global {r['packet_global_idx']})")
            print(f"       file: {r['file']}")
            print(f"       prev: {r['prev_time_ms']:.0f} ms ({prev_cfg}) -> curr: {r['curr_time_ms']:.0f} ms ({r['config']})")

        # Save to CSV
        out_dir = os.path.join(WORKSPACE, "results")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "time_reset_locations.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["#", "distance", "file", "config", "packet_in_file", "packet_global_idx", "prev_time_ms", "curr_time_ms", "prev_config"])
            for i, r in enumerate(reset_locations, 1):
                w.writerow([i, r["distance"], r["file"], r["config"], r["packet_in_file"], r["packet_global_idx"],
                            f"{r['prev_time_ms']:.0f}", f"{r['curr_time_ms']:.0f}", r.get("prev_config", "")])
        print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
