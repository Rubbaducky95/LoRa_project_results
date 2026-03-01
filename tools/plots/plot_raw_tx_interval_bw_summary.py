import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt


DATA_ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
OUTPUT_PNG = r"C:\Users\ruben\Documents\LoRa Project\results\raw_test_data_plots\raw_tx_interval_bw_summary.png"

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
            "lines.linewidth": 2.8,
            "lines.markersize": 6,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def mean_tx_interval(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8")))
    vals = []
    for r in rows[2:]:
        if len(r) < 5:
            continue
        try:
            if r[4]:
                vals.append(float(r[4]))
        except Exception:
            pass
    if not vals:
        return None
    return sum(vals) / len(vals)


def collect_summary():
    # summary[(bw, sf)] -> list of mean intervals across all distances and TPs
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
                if bw not in BWS or tp not in TP_VALUES or sf not in SF_VALUES:
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
    plt.figure(figsize=(8, 5))

    for bw in BWS:
        xs = []
        ys = []
        for sf in SF_VALUES:
            vals = summary.get((bw, sf), [])
            if not vals:
                continue
            xs.append(sf)
            ys.append(sum(vals) / len(vals))
        if xs:
            plt.plot(xs, ys, marker="o", linewidth=3.0, label=f"BW {bw}")

    plt.xticks(SF_VALUES)
    plt.xlabel("Spreading Factor (SF)")
    plt.ylabel("Average TX interval (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

