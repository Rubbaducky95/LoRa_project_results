"""
Plot average PER of all configurations over distance, with std dev and outliers at each distance.
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
            "lines.linewidth": 2,
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


def iqr_outliers(values):
    """Return indices of values that are outliers (IQR method)."""
    arr = np.array(values)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    if iqr == 0:
        return []
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return [i for i, v in enumerate(values) if v < lower or v > upper]


def collect_per_by_distance(data_root):
    """Returns per_by_distance: {distance: [per, per, ...]} for all configs."""
    per_by_distance = defaultdict(list)
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
                if (sf, bw, tp) == OUTLIER_CONFIG:
                    continue
                per_by_distance[distance].append(per)
    return dict(per_by_distance)


def main():
    parser = argparse.ArgumentParser(
        description="Plot average PER over distance with std dev and outliers."
    )
    parser.add_argument("--data-root", default=DATA_ROOT, help="Dataset root.")
    parser.add_argument("--output-dir", default=None, help="Output directory for PNGs.")
    args = parser.parse_args()
    if args.output_dir is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        args.output_dir = os.path.join(WORKSPACE, "results", out_dir)

    setup_plot_style()
    os.makedirs(args.output_dir, exist_ok=True)
    per_by_distance = collect_per_by_distance(args.data_root)

    if not per_by_distance:
        print("No data found.")
        return

    distances = sorted(per_by_distance.keys())
    avgs_pct = []
    stds_pct = []
    outlier_xs = []
    outlier_ys = []

    for d in distances:
        vals = per_by_distance[d]
        vals_pct = [v * 100 for v in vals]
        avgs_pct.append(np.mean(vals_pct))
        stds_pct.append(np.std(vals_pct) if len(vals_pct) > 1 else 0)
        out_idx = iqr_outliers(vals_pct)
        for i in out_idx:
            outlier_xs.append(d)
            outlier_ys.append(vals_pct[i])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(
        distances,
        avgs_pct,
        yerr=stds_pct,
        fmt="o-",
        color="#1f77b4",
        linewidth=2,
        markersize=6,
        capsize=4,
        capthick=1.5,
        label="Average ± std dev",
    )
    if outlier_xs:
        ax.scatter(
            outlier_xs,
            outlier_ys,
            color="#d62728",
            s=50,
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
            label="Outliers",
        )
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Average PER (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ticks = [t for t in XTICK_DISTANCES if min(distances) - 1e-6 <= t <= max(distances) + 1e-6]
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.2f}".rstrip("0").rstrip(".") for t in ticks], rotation=45)

    fig.tight_layout()
    prefix = "dataset" if "dataset" in args.data_root else "raw"
    out_path = os.path.join(args.output_dir, f"{prefix}_per_vs_distance_aggregated.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {prefix}_per_vs_distance_aggregated.png")


if __name__ == "__main__":
    main()
