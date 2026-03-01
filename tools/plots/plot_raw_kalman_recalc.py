import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATA_ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
OUTPUT_PNG = r"C:\Users\ruben\Documents\LoRa Project\results\raw_test_data_plots\raw_kalman_recalc_rssi_plots.png"

SF_VALUES = [7, 8, 9, 10, 11, 12]
TP_VALUES = [2, 12, 22]
BWS_TO_PLOT = [62500, 125000, 250000, 500000]


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
            "lines.linewidth": 2.4,
            "lines.markersize": 5,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
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


def mean_metric_from_file(path, metric_col):
    rows = list(csv.reader(open(path, "r", encoding="utf-8")))
    if not rows or metric_col not in rows[0]:
        return None
    idx = rows[0].index(metric_col)
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


def collect_data():
    # data[bw][sf][tp] = list[(distance, mean_metric)]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    metric_col = "kalman_rssi_recalc"

    for dn in sorted(os.listdir(DATA_ROOT)):
        folder = os.path.join(DATA_ROOT, dn)
        if not (os.path.isdir(folder) and dn.startswith("distance_")):
            continue
        distance = parse_distance(dn)
        if distance is None:
            continue

        for walk_root, _, files in os.walk(folder):
            for fn in files:
                cfg = parse_cfg(fn)
                if cfg is None:
                    continue
                sf, bw, tp = cfg
                if tp not in TP_VALUES or bw not in BWS_TO_PLOT:
                    continue
                mean_v = mean_metric_from_file(os.path.join(walk_root, fn), metric_col)
                if mean_v is None:
                    continue
                data[bw][sf][tp].append((distance, mean_v))

    for bw in data:
        for sf in data[bw]:
            for tp in data[bw][sf]:
                data[bw][sf][tp].sort(key=lambda x: x[0])
    return data


def main():
    setup_plot_style()
    data = collect_data()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    n_bw = len(BWS_TO_PLOT)
    fig, axes = plt.subplots(n_bw, 1, figsize=(11, 4.2 * n_bw), sharex=True, sharey=True)
    if n_bw == 1:
        axes = [axes]
    color_map = {sf: plt.cm.tab10(i % 10) for i, sf in enumerate(SF_VALUES)}
    tp_style = {
        2: {"linewidth": 1.8, "alpha": 0.45, "linestyle": "--"},
        12: {"linewidth": 3.0, "alpha": 1.00, "linestyle": "-"},
        22: {"linewidth": 1.8, "alpha": 0.45, "linestyle": ":"},
    }

    for row_idx, (ax, bw) in enumerate(zip(axes, BWS_TO_PLOT)):
        sf_series = data.get(bw, {})
        all_distances = sorted({d for sf in sf_series for tp in sf_series[sf] for d, _ in sf_series[sf][tp]})

        for sf in SF_VALUES:
            for tp in TP_VALUES:
                points = sf_series.get(sf, {}).get(tp, [])
                if not points:
                    continue
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                ax.plot(xs, ys, marker="o", markersize=5, color=color_map[sf], **tp_style[tp])

        if row_idx == len(BWS_TO_PLOT) - 1:
            ax.set_xlabel("Distance (m)")
        if all_distances:
            ax.set_xticks(all_distances)
            ax.set_xticklabels([f"{d:.2f}".rstrip("0").rstrip(".") for d in all_distances], rotation=45)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Average Kalman RSSI (recalc)")

    sf_handles = [Line2D([0], [0], color=color_map[sf], lw=2, label=f"SF{sf}") for sf in SF_VALUES]
    tp_handles = [Line2D([0], [0], color="black", label=f"TP={tp}", **tp_style[tp]) for tp in TP_VALUES]
    combined_handles = sf_handles + tp_handles
    combined_labels = [h.get_label() for h in combined_handles]
    for ax in axes:
        ax.legend(combined_handles, combined_labels, loc="upper right", ncol=3, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

