"""
Plot PER (%) vs distance for all configurations (SF, BW, TP).
Each point = one config at one distance. Color by BW.
"""
import argparse
import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
DATA_ROOT = os.path.join(WORKSPACE, "raw_test_data")

BW_VALUES = [62500, 125000, 250000, 500000]
TP_VALUES = [2, 12, 22]
SF_VALUES = [7, 8, 9, 10, 11, 12]
HEX_RE = re.compile(r"^[0-9A-F]+$")
XTICK_DISTANCES = [6.25, 25, 50, 75, 100]


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
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    return tuple(map(int, m.groups())) if m else None


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
    return (lost / total) * 100.0


def collect_all_configs(data_root):
    """Returns list of (distance, per, bw) for every config file."""
    points = []
    for dn in sorted(os.listdir(data_root)):
        dpath = os.path.join(data_root, dn)
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
                points.append((distance, per, bw))
    return points


def main():
    parser = argparse.ArgumentParser(
        description="Plot PER (%) vs distance for all configurations. Each point = one config."
    )
    parser.add_argument("--data-root", default=DATA_ROOT, help="Dataset root.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    args = parser.parse_args()
    if args.output is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        fname = "dataset_per_vs_distance_all_configs.png" if "dataset" in args.data_root else "raw_per_vs_distance_all_configs.png"
        args.output = os.path.join(WORKSPACE, "results", out_dir, fname)

    setup_plot_style()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    points = collect_all_configs(args.data_root)

    if not points:
        print("No data found.")
        return

    colors = {62500: "#9467bd", 125000: "#1f77b4", 250000: "#2ca02c", 500000: "#d62728"}

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for bw in BW_VALUES:
        pts = [(d, p) for d, p, b in points if b == bw]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=28, c=colors[bw], alpha=0.75, edgecolors="black", linewidths=0.3, label=f"BW {bw//1000} kHz")

    all_distances = sorted({p[0] for p in points})
    xmin, xmax = min(all_distances), max(all_distances)
    ticks = [d for d in XTICK_DISTANCES if xmin - 1e-6 <= d <= xmax + 1e-6]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{d:.2f}".rstrip("0").rstrip(".") for d in ticks], rotation=45)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel(r"PER (\%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
