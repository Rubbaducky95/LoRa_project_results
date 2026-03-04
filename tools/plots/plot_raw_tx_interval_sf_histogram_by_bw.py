"""
3D TX-interval histogram:
- x-axis: SF
- y-axis: BW
- z-axis: mean TX interval
- bar color: configurable extra metric
"""

import argparse
import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
from plot_config import FIGSIZE_ONE_COL, IEEE_FONTSIZE, SAVE_DPI, save_plot_outputs
from plot_rssi_vs_multiple import _add_rotated_rssi_scale


SF_VALUES = [7, 8, 9, 10, 11, 12]
BW_VALUES = [62500, 125000, 250000, 500000]
TP_VALUES = [2, 12, 22]

CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")
DIST_RE = re.compile(r"^distance_([\d.]+)m?$")

COLOR_METRICS = {
    "std": ("std_tx_interval_ms", r"Std. dev. (ms)", "cividis"),
    "range": ("range_tx_interval_ms", r"Range (ms)", "viridis"),
    "max": ("max_tx_interval_ms", r"Max TX interval (ms)", "magma"),
    "count": ("file_count", "Files", "plasma"),
}


def setup_plot_style():
    latex_ok = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "text.usetex": latex_ok,
            "font.size": IEEE_FONTSIZE,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.8,
            "axes.labelsize": IEEE_FONTSIZE,
            "xtick.labelsize": IEEE_FONTSIZE,
            "ytick.labelsize": IEEE_FONTSIZE,
        }
    )


def parse_cfg(filename):
    match = CFG_RE.match(filename)
    if not match:
        return None
    return tuple(map(int, match.groups()))


def parse_distance(folder_name):
    match = DIST_RE.match(folder_name)
    return float(match.group(1)) if match else None


def _tx_interval_col(header):
    for column in ("tx_interval_ms", "time_between_messages_ms"):
        if column in header:
            return header.index(column)
    return None


def _iter_packet_tx_intervals(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows:
        return []

    header = rows[0]
    tx_idx = _tx_interval_col(header)
    if tx_idx is None:
        return []
    payload_idx = header.index("payload") if "payload" in header else None

    values = []
    for row in rows[1:]:
        if len(row) <= tx_idx:
            continue
        if payload_idx is not None and len(row) > payload_idx:
            payload = row[payload_idx].strip()
            if payload.startswith("CFG "):
                continue
        try:
            if row[tx_idx]:
                values.append(float(row[tx_idx]))
        except Exception:
            pass
    return values


def collect_tx_interval_file_stats(data_root):
    """Collect per-file TX interval stats for each SF/BW/TP/distance config."""
    records = []
    for distance_dir in sorted(os.listdir(data_root)):
        dir_path = os.path.join(data_root, distance_dir)
        if not (os.path.isdir(dir_path) and distance_dir.startswith("distance_")):
            continue
        distance = parse_distance(distance_dir)
        if distance is None:
            continue
        for root, _, files in os.walk(dir_path):
            for filename in files:
                cfg = parse_cfg(filename)
                if cfg is None:
                    continue
                sf, bw, tp = cfg
                if sf not in SF_VALUES or bw not in BW_VALUES or tp not in TP_VALUES:
                    continue
                file_path = os.path.join(root, filename)
                values = _iter_packet_tx_intervals(file_path)
                if not values:
                    continue
                values_arr = np.asarray(values, dtype=float)
                records.append(
                    {
                        "distance_m": distance,
                        "sf": sf,
                        "bw_hz": bw,
                        "tp_dbm": tp,
                        "mean_tx_interval_ms": float(np.mean(values_arr)),
                        "std_tx_interval_ms": float(np.std(values_arr)) if values_arr.size > 1 else 0.0,
                        "max_tx_interval_ms": float(np.max(values_arr)),
                    }
                )
    return records


def aggregate_tx_interval_by_sf_bw(file_records):
    """Aggregate TX interval metrics across distance and TP for each SF/BW pair."""
    grouped = defaultdict(list)
    for record in file_records:
        grouped[(record["sf"], record["bw_hz"])].append(record)

    rows = []
    for sf in SF_VALUES:
        for bw in BW_VALUES:
            bucket = grouped.get((sf, bw), [])
            if not bucket:
                continue
            file_means = np.asarray([row["mean_tx_interval_ms"] for row in bucket], dtype=float)
            file_maxes = np.asarray([row["max_tx_interval_ms"] for row in bucket], dtype=float)
            rows.append(
                {
                    "sf": sf,
                    "bw_hz": bw,
                    "mean_tx_interval_ms": float(np.mean(file_means)),
                    "std_tx_interval_ms": float(np.std(file_means)) if file_means.size > 1 else 0.0,
                    "range_tx_interval_ms": float(np.max(file_means) - np.min(file_means)) if file_means.size > 1 else 0.0,
                    "max_tx_interval_ms": float(np.max(file_maxes)),
                    "file_count": float(len(bucket)),
                }
            )
    return rows


def _norm_from_rows(rows, key):
    values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=float)
    if values.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def plot_tx_interval_3d_histogram(agg_rows, output_png, color_by="std"):
    if not agg_rows:
        return

    color_key, color_label, cmap_name = COLOR_METRICS[color_by]
    norm = _norm_from_rows(agg_rows, color_key)
    cmap = plt.get_cmap(cmap_name)
    row_lookup = {(row["sf"], row["bw_hz"]): row for row in agg_rows}

    fig = plt.figure(figsize=FIGSIZE_ONE_COL)
    ax = fig.add_subplot(111, projection="3d")

    dx = 0.62
    dy = 0.62
    z_max = 0.0
    for sf_idx, sf in enumerate(SF_VALUES):
        for bw_idx, bw in enumerate(BW_VALUES):
            row = row_lookup.get((sf, bw))
            if row is None:
                continue
            z_height = float(row["mean_tx_interval_ms"])
            z_max = max(z_max, z_height)
            color_value = float(row[color_key])
            ax.bar3d(
                sf_idx - dx / 2.0,
                bw_idx - dy / 2.0,
                0.0,
                dx,
                dy,
                z_height,
                color=cmap(norm(color_value)),
                edgecolor="black",
                linewidth=0.45,
                shade=True,
                alpha=0.97,
            )

    ax.set_xlabel("SF", labelpad=1.0)
    ax.set_ylabel("BW (kHz)", labelpad=2.0)
    ax.set_zlabel("TX interval (ms)", labelpad=1.6)
    ax.set_xticks(np.arange(len(SF_VALUES), dtype=float))
    ax.set_xticklabels([str(sf) for sf in SF_VALUES])
    ax.set_yticks(np.arange(len(BW_VALUES), dtype=float))
    ax.set_yticklabels([str(int(bw / 1000)) if bw != 62500 else "62.5" for bw in BW_VALUES])
    ax.set_zlim(0.0, z_max * 1.10 if z_max > 0 else 1.0)
    ax.tick_params(axis="x", labelsize=IEEE_FONTSIZE, pad=0.3)
    ax.tick_params(axis="y", labelsize=IEEE_FONTSIZE, pad=0.3)
    ax.tick_params(axis="z", labelsize=IEEE_FONTSIZE, pad=0.3)
    ax.view_init(elev=24, azim=-55)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.72))
    except Exception:
        pass
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.08)
        pane.set_edgecolor((0.35, 0.35, 0.35, 0.22))
    ax.grid(True, alpha=0.28)

    _add_rotated_rssi_scale(
        fig,
        ax,
        cmap,
        float(norm.vmin),
        float(norm.vmax),
        label=color_label,
        fontsize=IEEE_FONTSIZE,
        scale_graph_side="right",
        label_graph_side="left",
    )
    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.06)
    png_path, pdf_path = save_plot_outputs(fig, output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot a 3D TX-interval histogram over SF and BW.")
    parser.add_argument(
        "--data-root",
        default=os.path.join(WORKSPACE, "raw_test_data"),
        help="Dataset root (default: raw_test_data/).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: derived from --data-root).",
    )
    parser.add_argument(
        "--color-by",
        choices=sorted(COLOR_METRICS.keys()),
        default="std",
        help="Metric used to color the bars.",
    )
    args = parser.parse_args()

    if args.output is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        filename = "dataset_tx_interval_sf_histogram_by_bw.png" if "dataset" in args.data_root else "raw_tx_interval_sf_histogram_by_bw.png"
        args.output = os.path.join(WORKSPACE, "results", out_dir, filename)

    setup_plot_style()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    file_records = collect_tx_interval_file_stats(args.data_root)
    agg_rows = aggregate_tx_interval_by_sf_bw(file_records)
    plot_tx_interval_3d_histogram(agg_rows, args.output, color_by=args.color_by)


if __name__ == "__main__":
    main()
