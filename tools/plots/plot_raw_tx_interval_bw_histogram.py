import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
OUTPUT_PNG = r"C:\Users\ruben\Documents\LoRa Project\results\raw_test_data_plots\raw_tx_interval_bw_histogram.png"

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
            "axes.titlesize": 13,
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


def mean_tx_interval(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows:
        return None
    header = rows[0]
    if "time_between_messages_ms" not in header:
        return None
    idx = header.index("time_between_messages_ms")
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


def collect_summary():
    # summary[(bw, sf)] -> list of file-level mean intervals across all distances and TPs
    summary = defaultdict(list)
    for dn in sorted(os.listdir(DATA_ROOT)):
        dpath = os.path.join(DATA_ROOT, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        for root, _, files in os.walk(dpath):
            for fn in files:
                cfg = parse_cfg(fn)
                if cfg is None:
                    continue
                sf, bw, tp = cfg
                if bw not in BWS or sf not in SF_VALUES or tp not in TP_VALUES:
                    continue
                m = mean_tx_interval(os.path.join(root, fn))
                if m is None:
                    continue
                summary[(bw, sf)].append(m)
    return summary


def main():
    setup_plot_style()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    summary = collect_summary()

    x = np.arange(len(SF_VALUES), dtype=float)
    n_bw = len(BWS)
    group_width = 0.82
    bar_width = group_width / n_bw
    start = -group_width / 2 + bar_width / 2
    colors = {62500: "#9467bd", 125000: "#1f77b4", 250000: "#2ca02c", 500000: "#d62728"}

    plt.figure(figsize=(10, 5.4))
    for i, bw in enumerate(BWS):
        ys = []
        for sf in SF_VALUES:
            vals = summary.get((bw, sf), [])
            ys.append((sum(vals) / len(vals)) if vals else np.nan)
        xpos = x + (start + i * bar_width)
        plt.bar(
            xpos,
            ys,
            width=bar_width * 0.94,
            color=colors.get(bw, None),
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            label=f"BW {bw}",
        )

    plt.xticks(x, [str(sf) for sf in SF_VALUES])
    plt.xlabel("Spreading Factor (SF)")
    plt.ylabel("Average TX interval (ms)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(ncol=2, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

