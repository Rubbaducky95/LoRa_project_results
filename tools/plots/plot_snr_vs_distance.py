"""
Plot average SNR over distance (aggregated across all configs, or per config).
Requires snr_db column in raw dataset (run add_snr_to_raw_dataset.py first).
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
XTICK_DISTANCES = [6.25, 25, 50, 75, 100]
OUTLIER_CONFIG = (10, 62500, 22)


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
            "lines.linewidth": 2,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    return tuple(map(int, m.groups())) if m else None


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def file_avg_snr(path):
    """Returns average SNR (dB) for file, or None if no snr_db column."""
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "snr_db" not in rows[0]:
        return None
    header = rows[0]
    snr_idx = header.index("snr_db")
    payload_idx = header.index("payload") if "payload" in header else -1
    vals = []
    for r in rows[1:]:
        if len(r) <= snr_idx:
            continue
        if payload_idx >= 0 and len(r) > payload_idx and str(r[payload_idx]).strip().startswith("CFG "):
            continue
        try:
            v = float(r[snr_idx])
            vals.append(v)
        except (ValueError, TypeError):
            pass
    if not vals:
        return None
    return np.mean(vals)


def collect_snr_by_distance(data_root):
    """Returns {distance: [snr, snr, ...]} for all configs, excluding outlier."""
    by_dist = defaultdict(list)
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
                if cfg == OUTLIER_CONFIG:
                    continue
                snr = file_avg_snr(os.path.join(root, fn))
                if snr is not None:
                    by_dist[distance].append(snr)
    return dict(by_dist)


def main():
    parser = argparse.ArgumentParser(description="Plot average SNR over distance.")
    parser.add_argument("--data-root", default=DATA_ROOT, help="Dataset root.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    args = parser.parse_args()
    if args.output_dir is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        args.output_dir = os.path.join(WORKSPACE, "results", out_dir)

    setup_plot_style()
    os.makedirs(args.output_dir, exist_ok=True)
    by_dist = collect_snr_by_distance(args.data_root)

    if not by_dist:
        print("No data found. Run add_snr_to_raw_dataset.py first for raw data.")
        return

    distances = sorted(by_dist.keys())
    avgs = [np.mean(by_dist[d]) for d in distances]
    stds = [np.std(by_dist[d]) if len(by_dist[d]) > 1 else 0 for d in distances]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(
        distances,
        avgs,
        yerr=stds,
        fmt="o-",
        color="#1f77b4",
        linewidth=2,
        markersize=6,
        capsize=4,
        capthick=1.5,
        label="Average ± std dev",
    )
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("SNR (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ticks = [t for t in XTICK_DISTANCES if min(distances) - 1e-6 <= t <= max(distances) + 1e-6]
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.2f}".rstrip("0").rstrip(".") for t in ticks], rotation=45)

    fig.tight_layout()
    prefix = "dataset" if "dataset" in args.data_root else "raw"
    out_path = os.path.join(args.output_dir, f"{prefix}_snr_vs_distance.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {prefix}_snr_vs_distance.png")


if __name__ == "__main__":
    main()
