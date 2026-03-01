import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt


DATA_ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
OUTPUT_PNG = r"C:\Users\ruben\Documents\LoRa Project\results\raw_test_data_plots\raw_per_bw_summary_scatter.png"

BW_VALUES = [62500, 125000, 250000, 500000]
TP_VALUES = [2, 12, 22]
SF_VALUES = [7, 8, 9, 10, 11, 12]
HEX_RE = re.compile(r"^[0-9A-F]+$")


def setup_plot_style():
    latex_ok = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "text.usetex": latex_ok,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.8,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def payload_is_valid(payload):
    if payload == "PACKET_LOST":
        return False
    parts = (payload or "").strip('"').split(",")
    for part in parts:
        if part and HEX_RE.match(part) is None:
            return False
    return True


def file_per(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8")))
    total = 0
    lost = 0
    for r in rows[2:]:
        if len(r) < 6:
            continue
        total += 1
        if not payload_is_valid(r[5]):
            lost += 1
    if total == 0:
        return None
    return (lost / total) * 100.0


def collect_summary():
    # grouped[(distance, bw)] -> [per values from all SF/TP files]
    grouped = defaultdict(list)
    for dn in sorted(os.listdir(DATA_ROOT)):
        dpath = os.path.join(DATA_ROOT, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        distance = parse_distance(dn)
        if distance is None:
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
                if per is None:
                    continue
                grouped[(distance, bw)].append(per)

    # summary[bw] = [(distance, avg_per_all_tp_sf), ...]
    summary = defaultdict(list)
    for (distance, bw), vals in grouped.items():
        if vals:
            summary[bw].append((distance, sum(vals) / len(vals)))
    for bw in summary:
        summary[bw].sort(key=lambda x: x[0])
    return summary


def main():
    setup_plot_style()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    summary = collect_summary()

    plt.figure(figsize=(9, 5.2))
    colors = {
        62500: "#9467bd",
        125000: "#1f77b4",
        250000: "#2ca02c",
        500000: "#d62728",
    }

    all_distances = set()
    for bw in BW_VALUES:
        points = summary.get(bw, [])
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        all_distances.update(xs)
        plt.scatter(xs, ys, s=55, color=colors[bw], alpha=0.9, label=f"BW {bw}")

    d_sorted = sorted(all_distances)
    if d_sorted:
        plt.xticks(d_sorted, [f"{d:.2f}".rstrip("0").rstrip(".") for d in d_sorted], rotation=45)

    plt.xlabel("Distance (m)")
    plt.ylabel(r"Average PER (\%)")
    plt.grid(True, alpha=0.35)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

