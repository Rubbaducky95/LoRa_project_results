import argparse
import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"

SF_VALUES = [7, 8, 9, 10, 11, 12]
TP_VALUES = [2, 12, 22]
BWS = [62500, 125000, 250000, 500000]


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
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def _tx_interval_col(header):
    for c in ("tx_interval_ms", "time_between_messages_ms"):
        if c in header:
            return header.index(c)
    return None


def mean_tx_interval(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows:
        return None
    header = rows[0]
    idx = _tx_interval_col(header)
    if idx is None:
        return None
    vals = []
    for row in rows[1:]:
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


def max_tx_interval(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows:
        return None
    header = rows[0]
    idx = _tx_interval_col(header)
    if idx is None:
        return None
    vmax = None
    for row in rows[1:]:
        if len(row) <= idx:
            continue
        try:
            if row[idx]:
                v = float(row[idx])
                vmax = v if vmax is None else max(vmax, v)
        except Exception:
            pass
    return vmax


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def collect_summary(data_root):
    # summary[(sf, bw)] -> list of file-level means across all distances and TPs
    summary = defaultdict(list)
    # bounds[(sf, bw)] -> {"low": max interval at 6.25m, "high": max interval at 100m}
    bounds = defaultdict(lambda: {"low": None, "high": None})

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
                if bw not in BWS or sf not in SF_VALUES or tp not in TP_VALUES:
                    continue
                fpath = os.path.join(root, fn)
                m = mean_tx_interval(fpath)
                if m is None:
                    continue
                summary[(sf, bw)].append(m)
                mx = max_tx_interval(fpath)
                if mx is not None and distance is not None:
                    if abs(distance - 6.25) < 1e-9:
                        prev = bounds[(sf, bw)]["low"]
                        bounds[(sf, bw)]["low"] = mx if prev is None else max(prev, mx)
                    elif abs(distance - 100.0) < 1e-9:
                        prev = bounds[(sf, bw)]["high"]
                        bounds[(sf, bw)]["high"] = mx if prev is None else max(prev, mx)
    return summary, bounds


def main():
    parser = argparse.ArgumentParser(description="Plot avg TX interval vs BW, grouped by SF.")
    parser.add_argument(
        "--data-root",
        default=os.path.join(WORKSPACE, "raw_test_data"),
        help="Dataset root (raw_test_data or dataset).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: derived from --data-root).",
    )
    args = parser.parse_args()
    if args.output is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        fname = "dataset_tx_interval_sf_histogram_by_bw.png" if "dataset" in args.data_root else "raw_tx_interval_sf_histogram_by_bw.png"
        args.output = os.path.join(WORKSPACE, "results", out_dir, fname)

    setup_plot_style()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary, bounds = collect_summary(args.data_root)

    x = np.arange(len(BWS), dtype=float)
    n_sf = len(SF_VALUES)
    group_width = 0.84
    bar_width = group_width / n_sf
    start = -group_width / 2 + bar_width / 2
    palette = ["blue", "red", "orange", "purple", "green", "deepskyblue"]
    colors = {sf: palette[i % len(palette)] for i, sf in enumerate(SF_VALUES)}

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for i, sf in enumerate(SF_VALUES):
        ys = []
        yerr_low = []
        yerr_high = []
        for bw in BWS:
            vals = summary.get((sf, bw), [])
            if vals:
                y = sum(vals) / len(vals)
                ys.append(y)
                low_anchor = bounds[(sf, bw)]["low"]
                high_anchor = bounds[(sf, bw)]["high"]
                if low_anchor is None:
                    low_anchor = y
                if high_anchor is None:
                    high_anchor = y
                yerr_low.append(max(0.0, y - low_anchor))
                yerr_high.append(max(0.0, high_anchor - y))
            else:
                ys.append(np.nan)
                yerr_low.append(0.0)
                yerr_high.append(0.0)
        xpos = x + (start + i * bar_width)
        bars = ax.bar(
            xpos,
            ys,
            width=bar_width * 0.92,
            color=colors[sf],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.45,
            label=f"SF{sf}",
        )
        # Asymmetric error bars:
        # - lower anchor: max tx interval at 6.25m
        # - upper anchor: max tx interval at 100m
        yerr = np.array([yerr_low, yerr_high], dtype=float)
        ax.errorbar(
            xpos,
            ys,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=2.5,
            zorder=4,
        )

    bw_labels = [str(int(bw / 1000)) for bw in BWS]
    ax.set_xticks(x, bw_labels)
    ax.set_xlabel("Bandwidth (kHz)")
    ax.set_ylabel("Average TX interval (ms)")
    ax.grid(True, axis="y", alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, ncol=1, framealpha=0.9)
    # Keep equal whitespace around the plotting area.
    fig.subplots_adjust(left=0.14, right=0.86, bottom=0.14, top=0.86)
    fig.savefig(args.output, dpi=220)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()

