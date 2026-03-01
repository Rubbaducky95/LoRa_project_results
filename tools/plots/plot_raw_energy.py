import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATA_ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
OUT_DISTANCE = r"C:\Users\ruben\Documents\LoRa Project\results\raw_test_data_plots\raw_energy_vs_distance.png"
OUT_SUMMARY = r"C:\Users\ruben\Documents\LoRa Project\results\raw_test_data_plots\raw_energy_summary.png"

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
            "lines.linewidth": 2.6,
            "lines.markersize": 5,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
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


def mean_energy_from_file(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8")))
    if not rows:
        return None
    header = rows[0]
    if "energy_per_packet_j" not in header:
        return None
    idx = header.index("energy_per_packet_j")
    vals = []
    for r in rows[2:]:
        if len(r) <= idx:
            continue
        try:
            if r[idx]:
                vals.append(float(r[idx]))
        except Exception:
            pass
    if not vals:
        return None
    return sum(vals) / len(vals)


def collect_data():
    # data[bw][sf][tp] = list[(distance, avg_energy_j)]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
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
                if sf not in SF_VALUES or bw not in BWS or tp not in TP_VALUES:
                    continue
                m = mean_energy_from_file(os.path.join(root, fn))
                if m is None:
                    continue
                data[bw][sf][tp].append((distance, m))
    for bw in data:
        for sf in data[bw]:
            for tp in data[bw][sf]:
                data[bw][sf][tp].sort(key=lambda x: x[0])
    return data


def plot_distance(data):
    setup_plot_style()
    os.makedirs(os.path.dirname(OUT_DISTANCE), exist_ok=True)
    n_bw = len(BWS)
    fig, axes = plt.subplots(1, n_bw, figsize=(5.5 * n_bw, 5), sharey=True)
    if n_bw == 1:
        axes = [axes]
    color_map = {sf: plt.cm.tab10(i % 10) for i, sf in enumerate(SF_VALUES)}
    tp_style = {
        2: {"linewidth": 1.8, "alpha": 0.45, "linestyle": "--"},
        12: {"linewidth": 3.0, "alpha": 1.00, "linestyle": "-"},
        22: {"linewidth": 1.8, "alpha": 0.45, "linestyle": ":"},
    }

    for ax, bw in zip(axes, BWS):
        sf_series = data.get(bw, {})
        all_distances = sorted({d for sf in sf_series for tp in sf_series[sf] for d, _ in sf_series[sf][tp]})
        for sf in SF_VALUES:
            for tp in TP_VALUES:
                pts = sf_series.get(sf, {}).get(tp, [])
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, marker="o", markersize=5, color=color_map[sf], **tp_style[tp])

        ax.set_xlabel("Distance (m)")
        if all_distances:
            ax.set_xticks(all_distances)
            ax.set_xticklabels([f"{d:.2f}".rstrip("0").rstrip(".") for d in all_distances], rotation=45)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Average energy per packet (J)")
    sf_handles = [Line2D([0], [0], color=color_map[sf], lw=2, label=f"SF{sf}") for sf in SF_VALUES]
    tp_handles = [Line2D([0], [0], color="black", label=f"TP={tp}", **tp_style[tp]) for tp in TP_VALUES]
    fig.legend(sf_handles, [h.get_label() for h in sf_handles], loc="upper center", bbox_to_anchor=(0.35, 0.955), ncol=6, frameon=False)
    fig.legend(tp_handles, [h.get_label() for h in tp_handles], loc="upper center", bbox_to_anchor=(0.82, 0.955), ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT_DISTANCE, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {OUT_DISTANCE}")


def plot_summary(data):
    setup_plot_style()
    # Build summary[tp][bw][sf] = mean over distance
    summary = defaultdict(lambda: defaultdict(dict))
    for bw in BWS:
        for sf in SF_VALUES:
            for tp in TP_VALUES:
                pts = data.get(bw, {}).get(sf, {}).get(tp, [])
                if not pts:
                    continue
                summary[tp][bw][sf] = sum(y for _, y in pts) / len(pts)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    color_by_bw = {62500: "#9467bd", 125000: "#1f77b4", 250000: "#2ca02c", 500000: "#d62728"}
    marker_by_bw = {62500: "D", 125000: "o", 250000: "s", 500000: "^"}

    for ax, tp in zip(axes, TP_VALUES):
        for bw in BWS:
            xs = []
            ys = []
            for sf in SF_VALUES:
                v = summary[tp].get(bw, {}).get(sf)
                if v is None:
                    continue
                xs.append(sf)
                ys.append(v)
            if xs:
                ax.plot(
                    xs,
                    ys,
                    marker=marker_by_bw[bw],
                    linewidth=2.8,
                    color=color_by_bw[bw],
                    linestyle="-",
                    label=f"BW{bw}",
                )
        ax.set_xlabel("Spreading Factor (SF)")
        ax.set_xticks(SF_VALUES)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, loc="upper left")

    axes[0].set_ylabel("Average energy per packet (J)")
    fig.tight_layout()
    fig.savefig(OUT_SUMMARY, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {OUT_SUMMARY}")


def main():
    data = collect_data()
    plot_distance(data)
    plot_summary(data)


if __name__ == "__main__":
    main()

