import argparse
import csv
import os
import re
import shutil
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DIST_RE = re.compile(r"^distance_([\d.]+)m?$")
CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")
# X-axis ticks: start at 6.25, then every 25 m
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
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
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
    sf, bw, tp = map(int, m.groups())
    return sf, bw, tp


def mean_rssi_from_file(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8")))
    header = rows[0] if rows else []
    if not rows or ("rssi_corrected" not in header and "rssi" not in header):
        return None
    idx = header.index("rssi_corrected") if "rssi_corrected" in header else header.index("rssi")
    time_idx = header.index("time_between_messages_ms") if "time_between_messages_ms" in header else None
    payload_idx = None
    if "payload" in header:
        payload_idx = header.index("payload")
    elif "payload [FILLED_ROWS]" in header:
        payload_idx = header.index("payload [FILLED_ROWS]")

    vals = []
    for row in rows[1:]:
        if len(row) <= idx:
            continue
        # Ignore config/ACK row when present (e.g., "CFG sf=...").
        if payload_idx is not None and len(row) > payload_idx:
            if str(row[payload_idx]).strip().startswith("CFG "):
                continue
        if time_idx is not None and len(row) > time_idx:
            if str(row[time_idx]).strip() == "":
                continue
        try:
            if row[idx]:
                vals.append(float(row[idx]))
        except Exception:
            pass
    if not vals:
        return None
    return sum(vals) / len(vals)


def collect_distance_tp_stats(data_root, tp_values):
    # grouped[(tp, distance)] -> [mean_rssi_per_config_file]
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
                _, _, tp = cfg
                if tp not in tp_values:
                    continue
                mean_rssi = mean_rssi_from_file(os.path.join(root, fn))
                if mean_rssi is None:
                    continue
                grouped[(tp, distance)].append(mean_rssi)

    # stats[tp] = list[(distance, mean, std, min, max)]
    stats = defaultdict(list)
    for (tp, distance), values in grouped.items():
        mu = sum(values) / len(values)
        sigma = statistics.stdev(values) if len(values) > 1 else 0.0
        vmin = min(values)
        vmax = max(values)
        stats[tp].append((distance, mu, sigma, vmin, vmax))

    for tp in stats:
        stats[tp].sort(key=lambda x: x[0])

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Plot average RSSI vs distance for each TP with std-dev error bars over SF/BW configs."
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join(WORKSPACE, "raw_test_data"),
        help="Dataset root (distance_* folders).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG file path (default: derived from --data-root).",
    )
    parser.add_argument("--tp-values", default="2,12,22", help="Comma-separated TP values.")
    parser.add_argument(
        "--single-plot",
        action="store_true",
        help="Plot all TP curves in one axis instead of separate subplots.",
    )
    args = parser.parse_args()
    if args.output is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        base = "dataset" if "dataset" in args.data_root else "raw"
        fname = f"{base}_avg_rssi_vs_distance_all_tp.png" if args.single_plot else f"{base}_avg_rssi_vs_distance_by_tp.png"
        args.output = os.path.join(WORKSPACE, "results", out_dir, fname)

    tp_values = [int(x.strip()) for x in args.tp_values.split(",") if x.strip()]
    setup_plot_style()
    stats = collect_distance_tp_stats(args.data_root, tp_values)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    color_by_tp = {2: "#1f77b4", 12: "#2ca02c", 22: "#d62728"}

    if args.single_plot:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
        for tp in tp_values:
            points = stats.get(tp, [])
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            yerr = [p[2] for p in points]
            mins = [p[3] for p in points]
            maxs = [p[4] for p in points]
            color = color_by_tp.get(tp, "black")
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                fmt="o",
                linestyle=":",
                color=color,
                ecolor=color,
                linewidth=2.0,
                markersize=6,
                elinewidth=1.8,
                capsize=4,
                capthick=1.5,
                alpha=0.95,
                label=f"TP={tp}",
            )
            ax.scatter(xs, mins, s=36, marker="o", facecolors="none", edgecolors=color, linewidths=1.5, alpha=0.95)
            ax.scatter(xs, maxs, s=36, marker="o", facecolors="none", edgecolors=color, linewidths=1.5, alpha=0.95)

        all_xs = sorted({p[0] for tp in tp_values for p in stats.get(tp, [])})
        if all_xs:
            xmin, xmax = min(all_xs), max(all_xs)
            ticks = [d for d in XTICK_DISTANCES if xmin - 1e-6 <= d <= xmax + 1e-6]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{d:.2f}".rstrip("0").rstrip(".") for d in ticks], rotation=45)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel(r"Average RSSI over SF/BW (dBm, $\pm \sigma$)")
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.9)
    else:
        fig, axes = plt.subplots(1, len(tp_values), figsize=(5.5 * len(tp_values), 4.2), sharey=True)
        if len(tp_values) == 1:
            axes = [axes]

        for ax, tp in zip(axes, tp_values):
            points = stats.get(tp, [])
            if points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                yerr = [p[2] for p in points]
                mins = [p[3] for p in points]
                maxs = [p[4] for p in points]
                color = color_by_tp.get(tp, "black")
                ax.errorbar(
                    xs,
                    ys,
                    yerr=yerr,
                    fmt="o",
                    linestyle=":",
                    color=color,
                    ecolor=color,
                    linewidth=2.0,
                    markersize=6,
                    elinewidth=1.8,
                    capsize=4,
                    capthick=1.5,
                    alpha=0.95,
                )
                # Show min/max as hollow points (outlier envelope) at each distance.
                ax.scatter(xs, mins, s=36, marker="o", facecolors="none", edgecolors=color, linewidths=1.5, alpha=0.95)
                ax.scatter(xs, maxs, s=36, marker="o", facecolors="none", edgecolors=color, linewidths=1.5, alpha=0.95)
                xmin, xmax = min(xs), max(xs)
                ticks = [d for d in XTICK_DISTANCES if xmin - 1e-6 <= d <= xmax + 1e-6]
                ax.set_xticks(ticks)
                ax.set_xticklabels([f"{d:.2f}".rstrip("0").rstrip(".") for d in ticks], rotation=45)
            ax.set_xlabel("Distance (m)")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(r"Average RSSI over SF/BW (dBm, $\pm \sigma$)")

    fig.tight_layout()
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()

