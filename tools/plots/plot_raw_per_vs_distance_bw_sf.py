import csv
import os
import re
import shutil
from collections import defaultdict
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt


DATA_ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
OUTPUT_PNG = r"C:\Users\ruben\Documents\LoRa Project\results\raw_test_data_plots\raw_per_vs_distance_bw_sf.png"

BW_VALUES = [62500, 125000, 250000, 500000]
TP_VALUES = [2, 12, 22]
SF_VALUES = [7, 8, 9, 10, 11, 12]
HEX_RE = re.compile(r"^[0-9A-F]+$")


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
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9,
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


def payload_is_valid(payload):
    if payload == "PACKET_LOST":
        return False
    parts = (payload or "").strip('"').split(",")
    for part in parts:
        if part and HEX_RE.match(part) is None:
            return False
    return True


def file_per(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8")))
    total = 0
    lost = 0
    for r in rows[2:]:
        if len(r) < 6:
            continue
        total += 1
        if not payload_is_valid(r[5]):
            lost += 1
    if total == 0:
        return None
    return (lost / total) * 100.0


def collect_summary():
    # grouped[(bw, sf, distance)] -> [per over TP]
    grouped = defaultdict(list)
    for dn in sorted(os.listdir(DATA_ROOT)):
        dpath = os.path.join(DATA_ROOT, dn)
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
                grouped[(bw, sf, distance)].append(per)

    # summary[bw][sf] = [(distance, avg_per_over_tp), ...]
    summary = defaultdict(lambda: defaultdict(list))
    for (bw, sf, distance), vals in grouped.items():
        if vals:
            summary[bw][sf].append((distance, sum(vals) / len(vals)))
    for bw in summary:
        for sf in summary[bw]:
            summary[bw][sf].sort(key=lambda x: x[0])
    return summary


def main():
    setup_plot_style()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    summary = collect_summary()
    color_map = {sf: plt.cm.tab10(i % 10) for i, sf in enumerate(SF_VALUES)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    # Jitter SF series slightly on x to reduce overlap at same distance.
    sf_offsets = {sf: (i - (len(SF_VALUES) - 1) / 2) * 0.18 for i, sf in enumerate(SF_VALUES)}

    for ax, bw in zip(axes, BW_VALUES):
        bw_data = summary.get(bw, {})
        all_distances = sorted({d for sf in bw_data for d, _ in bw_data[sf]})

        for sf in SF_VALUES:
            # Requested: ignore obvious SF10 outlier at BW=62500 for readability.
            if bw == 62500 and sf == 10:
                continue
            pts = bw_data.get(sf, [])
            if not pts:
                continue
            xs = [p[0] + sf_offsets[sf] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, s=38, color=color_map[sf], alpha=0.75, label=f"SF{sf}")

        ax.set_xlabel("Distance (m)")
        ax.set_ylabel(r"PER (\%)")
        if all_distances:
            ax.set_xticks(all_distances)
            ax.set_xticklabels([f"{d:.2f}".rstrip("0").rstrip(".") for d in all_distances], rotation=45)
        ax.grid(True, alpha=0.35)
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="", color=color_map[sf], label=f"SF{sf}", markersize=7)
            for sf in SF_VALUES
            if not (bw == 62500 and sf == 10)
        ]
        ax.legend(handles=legend_handles, loc="upper right", ncol=2, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()

