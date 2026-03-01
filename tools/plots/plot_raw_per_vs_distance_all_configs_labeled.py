"""
Plot PER vs distance for all configurations (SF, BW, TP) in one window.
4 panels (one per BW), each with lines for every (SF, TP) config, labeled with text.
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
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


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
    return round((lost / total) * 100.0, 4)  # percent, 4 decimal places


def collect_per_config(data_root):
    """Returns config[(sf, bw, tp)] = [(distance, per), ...] sorted by distance."""
    config = defaultdict(list)
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
                config[(sf, bw, tp)].append((distance, per))
    for key in config:
        config[key].sort(key=lambda x: x[0])
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Plot PER vs distance for all configs with text labels."
    )
    parser.add_argument("--data-root", default=DATA_ROOT, help="Dataset root.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    args = parser.parse_args()
    if args.output is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        fname = "dataset_per_vs_distance_all_labeled.png" if "dataset" in args.data_root else "raw_per_vs_distance_all_labeled.png"
        args.output = os.path.join(WORKSPACE, "results", out_dir, fname)

    setup_plot_style()
    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)
    config = collect_per_config(args.data_root)

    # Write PER values list (4 decimal places)
    csv_path = args.output.replace(".png", "_per_values.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["SF", "BW", "TP", "distance_m", "PER_pct"])
        for (sf, bw, tp) in sorted(config.keys()):
            for distance, per in config[(sf, bw, tp)]:
                w.writerow([sf, bw, tp, distance, f"{per:.4f}"])
    print(f"Saved PER values: {csv_path}")

    # Colors by SF, linestyles by TP
    colors = {sf: plt.cm.tab10(i % 10) for i, sf in enumerate(SF_VALUES)}
    ls_map = {2: "--", 12: "-", 22: ":"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, bw in enumerate(BW_VALUES):
        ax = axes[i]
        count = 0
        for sf in SF_VALUES:
            for tp in TP_VALUES:
                pts = config.get((sf, bw, tp), [])
                if not pts:
                    continue
                xs = np.array([p[0] for p in pts])
                ys = np.array([p[1] for p in pts])
                label = f"SF{sf}-T{tp}"
                ax.plot(
                    xs, ys,
                    color=colors[sf],
                    linestyle=ls_map[tp],
                    linewidth=1.5,
                    alpha=0.9,
                )
                # Label at last point, nudged right to avoid overlap
                x_last, y_last = xs[-1], ys[-1]
                ax.annotate(
                    label,
                    xy=(x_last, y_last),
                    xytext=(4, 0),
                    textcoords="offset points",
                    fontsize=6,
                    color=colors[sf],
                    alpha=0.95,
                    va="center",
                )
                count += 1

        ax.set_xlabel("Distance (m)")
        ax.set_ylabel(r"PER (\%)")
        ax.set_ylim(0, 7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda x, _: str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:.4f}"
        ))
        ax.set_title(f"BW {bw//1000} kHz", fontsize=10)
        ax.grid(True, alpha=0.3)

        all_distances = sorted({p[0] for (s, b, t), pts in config.items() if b == bw for p in pts})
        if all_distances:
            xmin, xmax = min(all_distances), max(all_distances)
            ticks = [d for d in XTICK_DISTANCES if xmin - 1e-6 <= d <= xmax + 1e-6]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{d:.2f}".rstrip("0").rstrip(".") for d in ticks], rotation=45)

    # Legend: SF colors + TP linestyles
    from matplotlib.lines import Line2D
    sf_handles = [Line2D([0], [0], color=colors[sf], lw=2, label=f"SF{sf}") for sf in SF_VALUES]
    tp_handles = [Line2D([0], [0], color="gray", ls=ls_map[tp], lw=2, label=f"TP{tp}") for tp in TP_VALUES]
    all_handles = sf_handles + tp_handles
    fig.legend(all_handles, [h.get_label() for h in all_handles], loc="upper center", ncol=9, framealpha=0.9, fontsize=8, bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
