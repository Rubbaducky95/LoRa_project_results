import argparse
import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Greyscale with slight offset on low end: low values = dark grey, high values = black
GREY_OFFSET_CMAP = LinearSegmentedColormap.from_list(
    "grey_offset", [(0.9, 0.9, 0.9), (0, 0, 0)]
)


CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")

SF_VALUES = [7, 8, 9, 10, 11, 12]
TP_VALUES = [2, 12, 22]
BW_VALUES = [62500, 125000, 250000, 500000]


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


def parse_cfg(filename):
    m = CFG_RE.match(filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None


def midpoint_energy_mj_for_file(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows:
        return None
    header = rows[0]

    # Support both naming variants.
    if "energy_per_packet_min_mj" in header and "energy_per_packet_max_mj" in header:
        i_min = header.index("energy_per_packet_min_mj")
        i_max = header.index("energy_per_packet_max_mj")
        scale = 1.0
    elif "energy_per_packet_j_min" in header and "energy_per_packet_j_max" in header:
        i_min = header.index("energy_per_packet_j_min")
        i_max = header.index("energy_per_packet_j_max")
        scale = 1000.0  # convert J -> mJ
    else:
        return None

    mids_raw = []
    for row in rows[1:]:
        if len(row) <= max(i_min, i_max):
            continue
        e_min = parse_float(row[i_min])
        e_max = parse_float(row[i_max])
        if e_min is None or e_max is None:
            continue
        mids_raw.append((e_min + e_max) / 2.0)

    if not mids_raw:
        return None

    # Some raw files use *_mj headers but store Joules (e.g. ~0.04).
    # Detect and correct by converting to mJ for plotting consistency.
    if scale == 1.0:
        sample = mids_raw[: min(20, len(mids_raw))]
        sample_avg = sum(sample) / len(sample)
        if sample_avg < 1.0:
            scale = 1000.0

    mids_mj = [m * scale for m in mids_raw]
    return sum(mids_mj) / len(mids_mj)


def collect_data(data_root):
    # acc[(tp,bw,sf)] -> [file-level avg midpoint energies over distances]
    acc = defaultdict(list)
    for dn in sorted(os.listdir(data_root)):
        dpath = os.path.join(data_root, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        for root, _, files in os.walk(dpath):
            for fn in files:
                cfg = parse_cfg(fn)
                if cfg is None:
                    continue
                sf, bw, tp = cfg
                if sf not in SF_VALUES or bw not in BW_VALUES or tp not in TP_VALUES:
                    continue
                v = midpoint_energy_mj_for_file(os.path.join(root, fn))
                if v is None:
                    continue
                acc[(tp, bw, sf)].append(v)

    # mean map
    mean_map = {}
    for key, vals in acc.items():
        mean_map[key] = sum(vals) / len(vals)
    return mean_map


def plot(mean_map, output_png, cmap="gray"):
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    # Build matrices TP-wise with rows=BW, cols=SF.
    mats = {}
    all_vals = []
    for tp in TP_VALUES:
        mat = np.full((len(BW_VALUES), len(SF_VALUES)), np.nan)
        for i_bw, bw in enumerate(BW_VALUES):
            for i_sf, sf in enumerate(SF_VALUES):
                v = mean_map.get((tp, bw, sf))
                if v is None:
                    continue
                mat[i_bw, i_sf] = v
                all_vals.append(v)
        mats[tp] = mat

    if not all_vals:
        raise RuntimeError("No usable min/max energy values found in dataset.")

    vmin, vmax = min(all_vals), max(all_vals)
    fig, axes = plt.subplots(3, 1, figsize=(4.4, 6.4), sharex=True, sharey=True, constrained_layout=False)

    image = None
    for i, (ax, tp) in enumerate(zip(axes, TP_VALUES)):
        mat = mats[tp]
        image = ax.imshow(
            mat,
            aspect="equal",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_box_aspect(len(BW_VALUES) / len(SF_VALUES))
        ax.set_xticks(range(len(SF_VALUES)))
        ax.set_xticklabels([str(x) for x in SF_VALUES])
        ax.set_yticks(range(len(BW_VALUES)))
        ax.set_yticklabels([f"{x / 1000:.1f}".rstrip("0").rstrip(".") for x in BW_VALUES])
        if i < len(TP_VALUES) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel("SF")
        # Keep TP identification without plot title.
        ax.text(
            0.98,
            0.98,
            f"TP={tp}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
        )
        # Annotate each cell with avg midpoint energy (mJ).
        for i_bw in range(len(BW_VALUES)):
            for i_sf in range(len(SF_VALUES)):
                val = mat[i_bw, i_sf]
                if np.isnan(val):
                    continue
                frac = 0.5 if vmax <= vmin else (val - vmin) / (vmax - vmin)
                txt_color = "white" if frac >= 0.6 else "black"
                ax.text(
                    i_sf,
                    i_bw,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=txt_color,
                )

    axes[1].set_ylabel("BW (kHz)")
    fig.subplots_adjust(left=0.16, right=0.82, top=0.99, bottom=0.06, hspace=-0.0)
    cax = fig.add_axes([0.84, 0.063, 0.018, 0.922])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Avg packet energy (mJ)")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {output_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot avg midpoint packet energy heatmaps (BW vs SF) for each TP."
    )
    parser.add_argument(
        "--data-root",
        default=r"C:\Users\ruben\Documents\LoRa Project\dataset",
        help="Dataset root to read (raw_test_data or dataset).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG base path (default: derived from --data-root).",
    )
    args = parser.parse_args()
    if args.output is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        fname = "dataset_energy_minmax_gradient_by_tp.png" if "dataset" in args.data_root else "raw_energy_minmax_gradient_by_tp.png"
        args.output = os.path.join(r"C:\Users\ruben\Documents\LoRa Project", "results", out_dir, fname)

    setup_plot_style()
    mean_map = collect_data(args.data_root)
    base = args.output
    if base.lower().endswith(".png"):
        base = base[:-4]
    plot(mean_map, f"{base}_greyscale.png", cmap=GREY_OFFSET_CMAP)
    plot(mean_map, f"{base}_color.png", cmap="hot_r")


if __name__ == "__main__":
    main()
