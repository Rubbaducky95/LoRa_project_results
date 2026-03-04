"""
Plot PER in 4 quadrants (one per BW). Center = 0% PER, radius outward = 100% PER.
Distance maps to angle within each quadrant.
"""
import argparse
import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_ROOT = os.path.join(WORKSPACE, "raw_test_data")
OUTPUT_PNG = os.path.join(WORKSPACE, "results", "raw_test_data_plots", "raw_per_quadrant_by_bw.png")

BW_VALUES = [62500, 125000, 250000, 500000]
TP_VALUES = [2, 12, 22]
SF_VALUES = [7, 8, 9, 10, 11, 12]
HEX_RE = re.compile(r"^[0-9A-F]+$")
DIST_MIN, DIST_MAX = 6.25, 100.0


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
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
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


def collect_summary(data_root):
    grouped = defaultdict(list)
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
                grouped[(distance, bw)].append(per)

    summary = defaultdict(list)
    for (distance, bw), vals in grouped.items():
        if vals:
            summary[bw].append((distance, sum(vals) / len(vals)))
    for bw in summary:
        summary[bw].sort(key=lambda x: x[0])
    return summary


def main():
    parser = argparse.ArgumentParser(description="Plot PER in 4 quadrants (one per BW), center=0%.")
    parser.add_argument("--data-root", default=DATA_ROOT, help="Dataset root.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    args = parser.parse_args()
    if args.output is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        fname = "dataset_per_quadrant_by_bw.png" if "dataset" in args.data_root else "raw_per_quadrant_by_bw.png"
        args.output = os.path.join(WORKSPACE, "results", out_dir, fname)

    setup_plot_style()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary = collect_summary(args.data_root)

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5), sharex=True, sharey=True, subplot_kw=dict(aspect="equal"))
    axes = axes.flatten()

    colors = {62500: "#9467bd", 125000: "#1f77b4", 250000: "#2ca02c", 500000: "#d62728"}

    for i, bw in enumerate(BW_VALUES):
        ax = axes[i]
        points = summary.get(bw, [])
        if not points:
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.text(50, 50, f"BW {bw//1000} kHz\n(no data)", ha="center", va="center", fontsize=10)
            ax.set_title(f"BW {bw//1000} kHz", fontsize=10)
            continue

        # Map (distance, per) to (x, y): center=0% PER, radius=PER, angle from distance
        xs, ys, pers, dists = [], [], [], []
        for distance, per in points:
            angle = np.pi / 2 * (distance - DIST_MIN) / (DIST_MAX - DIST_MIN) if DIST_MAX > DIST_MIN else 0
            r = per
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            xs.append(x)
            ys.append(y)
            pers.append(per)
            dists.append(distance)

        ax.scatter(xs, ys, s=50, c=colors[bw], alpha=0.9, edgecolors="black", linewidths=0.5)

        # Reference circles at 25, 50, 75, 100%
        for r in [25, 50, 75, 100]:
            theta = np.linspace(0, np.pi / 2, 50)
            ax.plot(r * np.cos(theta), r * np.sin(theta), "k-", alpha=0.25, linewidth=0.8)
        ax.plot([0, 100], [0, 0], "k-", alpha=0.2, linewidth=0.6)
        ax.plot([0, 0], [0, 100], "k-", alpha=0.2, linewidth=0.6)

        ax.set_xlim(0, 105)
        ax.set_ylim(0, 105)
        ax.set_title(f"BW {bw//1000} kHz", fontsize=10)

    for ax in axes:
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])

    fig.tight_layout()
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
