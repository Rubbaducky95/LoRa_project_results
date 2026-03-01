import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATA_ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
OUT_DIR = r"C:\Users\ruben\Documents\LoRa Project\results\raw_test_data_plots\per_vs_distance"

SF_VALUES = [7, 8, 9, 10, 11, 12]
TP_VALUES = [2, 12, 22]
BW_VALUES = [62500, 125000, 250000, 500000]
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
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
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


def collect_data():
    # data[(bw,tp)][sf] = [(distance, per_pct), ...]
    data = defaultdict(lambda: defaultdict(list))
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
                if sf not in SF_VALUES or bw not in BW_VALUES or tp not in TP_VALUES:
                    continue
                per = file_per(os.path.join(root, fn))
                if per is None:
                    continue
                data[(bw, tp)][sf].append((distance, per))
    for key in data:
        for sf in data[key]:
            data[key][sf].sort(key=lambda x: x[0])
    return data


def main():
    setup_plot_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    data = collect_data()
    color_map = {sf: plt.cm.tab10(i % 10) for i, sf in enumerate(SF_VALUES)}
    tp_style = {
        2: {"linestyle": "--", "alpha": 0.45, "linewidth": 1.8},
        12: {"linestyle": "-", "alpha": 1.00, "linewidth": 3.0},
        22: {"linestyle": ":", "alpha": 0.45, "linewidth": 1.8},
    }

    for bw in BW_VALUES:
        has_any = any(data.get((bw, tp), {}) for tp in TP_VALUES)
        if not has_any:
            continue

        plt.figure(figsize=(9.5, 5.5))
        all_distances = set()
        for tp in TP_VALUES:
            series = data.get((bw, tp), {})
            for sf in SF_VALUES:
                pts = series.get(sf, [])
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                all_distances.update(xs)
                plt.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=5,
                    color=color_map[sf],
                    **tp_style[tp],
                )

        distances = sorted(all_distances)
        if distances:
            plt.xticks(distances, [f"{d:.2f}".rstrip("0").rstrip(".") for d in distances], rotation=45)

        plt.xlabel("Distance (m)")
        plt.ylabel(r"Packet Error Rate (\%)")
        plt.grid(True, alpha=0.35)

        sf_handles = [Line2D([0], [0], color=color_map[sf], lw=2, label=f"SF{sf}") for sf in SF_VALUES]
        tp_handles = [Line2D([0], [0], color="black", label=f"TP={tp}", **tp_style[tp]) for tp in TP_VALUES]
        leg1 = plt.legend(sf_handles, [h.get_label() for h in sf_handles], loc="upper left", ncol=3, framealpha=0.9)
        plt.gca().add_artist(leg1)
        plt.legend(tp_handles, [h.get_label() for h in tp_handles], loc="upper right", ncol=1, framealpha=0.9)

        plt.tight_layout()
        stem = f"per_bw{bw:06d}_alltp"
        out_png = os.path.join(OUT_DIR, f"{stem}.png")
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

