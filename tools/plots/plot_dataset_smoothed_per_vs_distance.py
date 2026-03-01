import csv
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt


DATA_ROOT = r"C:\Users\ruben\Documents\LoRa Project\dataset_smoothed"
OUT_DIR = r"C:\Users\ruben\Documents\LoRa Project\results\dataset_plots\per_vs_distance_smoothed"

SF_VALUES = [7, 8, 9, 10, 11, 12]
TP_VALUES = [2, 12, 22]
BW_VALUES = [62500, 125000, 250000, 500000]
HEX_RE = re.compile(r"^[0-9A-F]+$")
CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


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
    ip = rows[0].index("payload")
    total = 0
    lost = 0
    for r in rows[1:]:
        if len(r) <= ip:
            continue
        total += 1
        if not payload_is_valid(r[ip]):
            lost += 1
    if total == 0:
        return None
    return (lost / total) * 100.0


def collect_data():
    data = defaultdict(lambda: defaultdict(list))  # data[(bw,tp)][sf] -> [(d,per)]
    for dn in sorted(os.listdir(DATA_ROOT)):
        dpath = os.path.join(DATA_ROOT, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        d = parse_distance(dn)
        if d is None:
            continue
        for wr, _, files in os.walk(dpath):
            for fn in files:
                m = CFG_RE.match(fn)
                if not m:
                    continue
                sf, bw, tp = map(int, m.groups())
                if sf not in SF_VALUES or bw not in BW_VALUES or tp not in TP_VALUES:
                    continue
                per = file_per(os.path.join(wr, fn))
                if per is None:
                    continue
                data[(bw, tp)][sf].append((d, per))
    for key in data:
        for sf in data[key]:
            data[key][sf].sort(key=lambda x: x[0])
    return data


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = collect_data()
    color_map = {sf: plt.cm.tab10(i % 10) for i, sf in enumerate(SF_VALUES)}
    tp_style = {
        2: {"linestyle": "--", "alpha": 0.45, "linewidth": 1.8},
        12: {"linestyle": "-", "alpha": 1.00, "linewidth": 3.0},
        22: {"linestyle": ":", "alpha": 0.45, "linewidth": 1.8},
    }

    for bw in BW_VALUES:
        if not any(data.get((bw, tp), {}) for tp in TP_VALUES):
            continue
        plt.figure(figsize=(9, 5))
        all_distances = set()
        for tp in TP_VALUES:
            series = data.get((bw, tp), {})
            for sf in SF_VALUES:
                pts = series.get(sf, [])
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                all_distances.update(xs)
                plt.plot(xs, ys, marker="o", markersize=4.5, color=color_map[sf], **tp_style[tp])
        d_sorted = sorted(all_distances)
        if d_sorted:
            plt.xticks(d_sorted, [f"{x:.2f}".rstrip("0").rstrip(".") for x in d_sorted], rotation=45)
        plt.xlabel("Distance (m)")
        plt.ylabel("PER (%)")
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, f"per_bw{bw:06d}_alltp.png")
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
