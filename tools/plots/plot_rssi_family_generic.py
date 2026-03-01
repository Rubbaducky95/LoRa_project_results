import argparse
import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DIST_RE = re.compile(r"^distance_([\d.]+)m?$")
CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")


def parse_int_list(text, default):
    if not text:
        return default
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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
    m = DIST_RE.match(folder_name)
    return float(m.group(1)) if m else None


def parse_cfg(filename):
    m = CFG_RE.match(filename)
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


def collect_data(data_root, metric_col, sf_values, bw_values, tp_values):
    # data[bw][sf][tp] = list[(distance, mean_metric)]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for dn in sorted(os.listdir(data_root)):
        folder = os.path.join(data_root, dn)
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
                if sf not in sf_values or bw not in bw_values or tp not in tp_values:
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
    parser = argparse.ArgumentParser(
        description="Generic RSSI-family plotter for distance_/SF*/CSV datasets."
    )
    parser.add_argument("--data-root", required=True, help="Dataset root (e.g. test_data2 or raw_test_data).")
    parser.add_argument("--metric-col", default="rssi_corrected", help="Column name to average and plot (rssi_corrected or rssi).")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--title", default="Metric vs Distance by SF")
    parser.add_argument("--ylabel", default="Average value")
    parser.add_argument("--sf-values", default="7,8,9,10,11,12", help="Comma-separated SF values.")
    parser.add_argument("--bw-values", default="62500,125000,250000,500000", help="Comma-separated BW values.")
    parser.add_argument("--tp-values", default="2,12,22", help="Comma-separated TP values.")
    parser.add_argument(
        "--legend-corner",
        default="upper right",
        choices=["upper right", "upper left", "lower right", "lower left"],
        help="Legend location inside each subplot.",
    )
    args = parser.parse_args()

    sf_values = parse_int_list(args.sf_values, [7, 8, 9, 10, 11, 12])
    bw_values = parse_int_list(args.bw_values, [125000, 250000, 500000])
    tp_values = parse_int_list(args.tp_values, [2, 12, 22])

    setup_plot_style()
    data = collect_data(args.data_root, args.metric_col, sf_values, bw_values, tp_values)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    nrows = len(bw_values)
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 4.6 * nrows), sharex=True, sharey=True)
    if nrows == 1:
        axes = [axes]

    color_map = {sf: plt.cm.tab10(i % 10) for i, sf in enumerate(sf_values)}
    tp_style = {
        tp_values[0]: {"linewidth": 1.8, "alpha": 0.45, "linestyle": "--"},
        tp_values[1] if len(tp_values) > 1 else tp_values[0]: {"linewidth": 3.0, "alpha": 1.00, "linestyle": "-"},
        tp_values[2] if len(tp_values) > 2 else tp_values[-1]: {"linewidth": 1.8, "alpha": 0.45, "linestyle": ":"},
    }

    for row_idx, (ax, bw) in enumerate(zip(axes, bw_values)):
        sf_series = data.get(bw, {})
        all_distances = sorted({d for sf in sf_series for tp in sf_series[sf] for d, _ in sf_series[sf][tp]})

        for sf in sf_values:
            for tp in tp_values:
                points = sf_series.get(sf, {}).get(tp, [])
                if not points:
                    continue
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                style = tp_style.get(tp, {"linewidth": 2.0, "alpha": 0.7, "linestyle": "-"})
                ax.plot(xs, ys, marker="o", markersize=5, color=color_map[sf], **style)

        if row_idx == nrows - 1:
            ax.set_xlabel("Distance (m)")
        if all_distances:
            ax.set_xticks(all_distances)
            ax.set_xticklabels([f"{d:.2f}".rstrip("0").rstrip(".") for d in all_distances], rotation=45)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(args.ylabel)

    sf_handles = [Line2D([0], [0], color=color_map[sf], lw=2, label=f"SF{sf}") for sf in sf_values]
    tp_handles = []
    for tp in tp_values:
        style = tp_style.get(tp, {"linewidth": 2.0, "alpha": 0.7, "linestyle": "-"})
        tp_handles.append(Line2D([0], [0], color="black", label=f"TP={tp}", **style))
    combined_handles = sf_handles + tp_handles
    combined_labels = [h.get_label() for h in combined_handles]
    for ax in axes:
        ax.legend(combined_handles, combined_labels, loc=args.legend_corner, ncol=3, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()

