"""
Plot PER histograms (ignoring distance): PER vs BW, PER vs SF, PER vs TP.
Aggregates all PER values across distances for each config.
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
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


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
    return lost / total  # fraction 0-1


OUTLIER_CONFIG = (10, 62500, 22)  # SF10, BW 62.5 kHz, TP 22


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def collect_all_per(data_root):
    """Returns per_by_bw, per_by_sf, per_by_tp (excluding outlier), and outlier [(distance, per), ...]."""
    per_by_bw = defaultdict(list)
    per_by_sf = defaultdict(list)
    per_by_tp = defaultdict(list)
    outlier_data = []  # [(distance, per), ...] for SF10, BW62500, TP22
    for dn in sorted(os.listdir(data_root)):
        dpath = os.path.join(data_root, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        distance = parse_distance(dn)
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
                if (sf, bw, tp) == OUTLIER_CONFIG:
                    if distance is not None:
                        outlier_data.append((distance, per))
                    continue
                per_by_bw[bw].append(per)
                per_by_sf[sf].append(per)
                per_by_tp[tp].append(per)
    outlier_data.sort(key=lambda x: x[0])
    return per_by_bw, per_by_sf, per_by_tp, outlier_data


def main():
    parser = argparse.ArgumentParser(
        description="Plot PER histograms vs BW, SF, TP (ignoring distance)."
    )
    parser.add_argument("--data-root", default=DATA_ROOT, help="Dataset root.")
    parser.add_argument("--output-dir", default=None, help="Output directory for PNGs.")
    args = parser.parse_args()
    if args.output_dir is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        args.output_dir = os.path.join(WORKSPACE, "results", out_dir)

    setup_plot_style()
    os.makedirs(args.output_dir, exist_ok=True)
    per_by_bw, per_by_sf, per_by_tp, outlier_data = collect_all_per(args.data_root)

    prefix = "dataset" if "dataset" in args.data_root else "raw"

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # PER vs BW
    ax = axes[0]
    bws = [bw for bw in BW_VALUES if per_by_bw[bw]]
    avgs = [np.mean(per_by_bw[bw]) * 100 for bw in bws]
    x = np.arange(len(bws))
    ax.bar(x, avgs, color="#1f77b4", alpha=0.8, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{bw//1000}" for bw in bws])
    ax.set_xlabel("Bandwidth (kHz)")
    ax.set_ylabel("Average PER (%)")
    ax.set_ylim(0, 0.4)
    ax.grid(True, axis="y", alpha=0.3)

    # PER vs SF
    ax = axes[1]
    sfs = [sf for sf in SF_VALUES if per_by_sf[sf]]
    avgs = [np.mean(per_by_sf[sf]) * 100 for sf in sfs]
    x = np.arange(len(sfs))
    ax.bar(x, avgs, color="#2ca02c", alpha=0.8, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([str(sf) for sf in sfs])
    ax.set_xlabel("Spreading Factor (SF)")
    ax.set_ylabel("Average PER (%)")
    ax.set_ylim(0, 0.4)
    ax.grid(True, axis="y", alpha=0.3)

    # PER vs TP
    ax = axes[2]
    tps = [tp for tp in TP_VALUES if per_by_tp[tp]]
    avgs = [np.mean(per_by_tp[tp]) * 100 for tp in tps]
    x = np.arange(len(tps))
    ax.bar(x, avgs, color="#d62728", alpha=0.8, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([str(tp) for tp in tps])
    ax.set_xlabel("Transmit Power (dBm)")
    ax.set_ylabel("Average PER (%)")
    ax.set_ylim(0, 0.4)
    ax.grid(True, axis="y", alpha=0.3)

    # Outlier: SF10, BW 62.5 kHz, TP 22
    ax = axes[3]
    if outlier_data:
        xs = [p[0] for p in outlier_data]
        ys = [p[1] * 100 for p in outlier_data]
        ax.plot(xs, ys, "o-", color="#9467bd", linewidth=2, markersize=5)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Average PER (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.set_title("SF10, BW 62.5 kHz, TP 22 dBm (outlier)")
        xmin, xmax = min(xs), max(xs)
        ticks = [d for d in XTICK_DISTANCES if xmin - 1e-6 <= d <= xmax + 1e-6]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{d:.2f}".rstrip("0").rstrip(".") for d in ticks], rotation=45)
    else:
        ax.text(0.5, 0.5, "No data for SF10, BW 62.5 kHz, TP 22", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("SF10, BW 62.5 kHz, TP 22 dBm (outlier)")

    fig.tight_layout()
    out_path = os.path.join(args.output_dir, f"{prefix}_per_histograms.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {prefix}_per_histograms.png")


if __name__ == "__main__":
    main()
