import argparse
import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


DIST_RE = re.compile(r"^distance_([\d.]+)m?$")
CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")
DEFAULT_EXCLUDED_CONFIGS = {(11, 62500), (12, 62500), (12, 125000)}


def setup_plot_style():
    latex_ok = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "text.usetex": latex_ok,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.2,
        }
    )


def parse_distance(folder_name):
    m = DIST_RE.match(folder_name)
    return float(m.group(1)) if m else None


def parse_cfg(filename):
    m = CFG_RE.match(filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def parse_excluded_configs(text):
    """
    Parse 'sf:bw,sf:bw,...' into a set of (sf, bw) tuples.
    Example: '11:62500,12:62500,12:125000'
    """
    excluded = set()
    if not text:
        return excluded
    for item in text.split(","):
        part = item.strip()
        if not part or ":" not in part:
            continue
        sf_txt, bw_txt = part.split(":", 1)
        try:
            excluded.add((int(sf_txt.strip()), int(bw_txt.strip())))
        except Exception:
            pass
    return excluded


def mean_rssi_from_file(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8")))
    if not rows or ("rssi_corrected" not in rows[0] and "rssi" not in rows[0]):
        return None
    idx = rows[0].index("rssi_corrected") if "rssi_corrected" in rows[0] else rows[0].index("rssi")
    vals = []
    for row in rows[2:]:
        if len(row) <= idx:
            continue
        try:
            if row[idx]:
                vals.append(float(row[idx]))
        except Exception:
            pass
    if not vals:
        return None
    return sum(vals) / len(vals)


def collect_means(data_root, tp_values, sf_values, bw_values, excluded_configs):
    # means[tp][(sf, bw)][distance] = mean_rssi
    means = defaultdict(lambda: defaultdict(dict))
    distances = set()

    for dn in sorted(os.listdir(data_root)):
        dpath = os.path.join(data_root, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        distance = parse_distance(dn)
        if distance is None:
            continue
        distances.add(distance)

        for root, _, files in os.walk(dpath):
            for fn in files:
                cfg = parse_cfg(fn)
                if cfg is None:
                    continue
                sf, bw, tp = cfg
                if tp not in tp_values or sf not in sf_values or bw not in bw_values:
                    continue
                if (sf, bw) in excluded_configs:
                    continue
                avg_rssi = mean_rssi_from_file(os.path.join(root, fn))
                if avg_rssi is None:
                    continue
                means[tp][(sf, bw)][distance] = avg_rssi

    return means, sorted(distances)


def main():
    parser = argparse.ArgumentParser(
        description="Plot average RSSI color-gradient maps by TP (configs vs distance)."
    )
    parser.add_argument(
        "--data-root",
        default=r"C:\Users\ruben\Documents\LoRa Project\raw_test_data",
        help="Dataset root with distance_* folders.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: derived from --data-root).",
    )
    parser.add_argument("--tp-values", default="2,12,22", help="Comma-separated TP list.")
    parser.add_argument("--sf-values", default="7,8,9,10,11,12", help="Comma-separated SF list.")
    parser.add_argument("--bw-values", default="62500,125000,250000,500000", help="Comma-separated BW list.")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name.")
    parser.add_argument(
        "--exclude-configs",
        default="11:62500,12:62500,12:125000",
        help="Comma-separated SF:BW pairs to exclude, e.g. '11:62500,12:62500,12:125000'.",
    )
    args = parser.parse_args()
    if args.output is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        fname = "dataset_avg_rssi_gradient_by_tp.png" if "dataset" in args.data_root else "raw_avg_rssi_gradient_by_tp.png"
        args.output = os.path.join(r"C:\Users\ruben\Documents\LoRa Project", "results", out_dir, fname)

    tp_values = [int(x.strip()) for x in args.tp_values.split(",") if x.strip()]
    sf_values = [int(x.strip()) for x in args.sf_values.split(",") if x.strip()]
    bw_values = [int(x.strip()) for x in args.bw_values.split(",") if x.strip()]
    excluded_configs = parse_excluded_configs(args.exclude_configs) or DEFAULT_EXCLUDED_CONFIGS

    setup_plot_style()
    means, distances = collect_means(args.data_root, tp_values, sf_values, bw_values, excluded_configs)
    if not distances:
        raise RuntimeError("No distance folders/data found for the selected dataset.")

    configs = [(sf, bw) for sf in sf_values for bw in bw_values if (sf, bw) not in excluded_configs]
    cfg_labels = [f"SF{sf}-BW{bw}" for sf, bw in configs]

    # Build matrices and global scale for consistent color comparison across TP panels.
    matrices = {}
    all_vals = []
    for tp in tp_values:
        mat = np.full((len(configs), len(distances)), np.nan, dtype=float)
        for i, cfg in enumerate(configs):
            for j, d in enumerate(distances):
                v = means.get(tp, {}).get(cfg, {}).get(d)
                if v is not None:
                    mat[i, j] = v
                    all_vals.append(v)
        matrices[tp] = mat

    if not all_vals:
        raise RuntimeError("No RSSI values found for selected TP/SF/BW filters.")

    vmin = min(all_vals)
    vmax = max(all_vals)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig, axes = plt.subplots(
        1, len(tp_values), figsize=(6.0 * len(tp_values), 8.5), sharey=True, constrained_layout=True
    )
    if len(tp_values) == 1:
        axes = [axes]

    image = None
    for ax, tp in zip(axes, tp_values):
        mat = matrices[tp]
        image = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=args.cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel("Distance (m)")
        ax.set_xticks(range(len(distances)))
        ax.set_xticklabels([f"{d:.2f}".rstrip("0").rstrip(".") for d in distances], rotation=45, ha="right")
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(cfg_labels)

    axes[0].set_ylabel("Configuration (SF, BW)")
    cbar = fig.colorbar(image, ax=axes, shrink=0.95, pad=0.02)
    cbar.set_label("Average RSSI")
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()

