"""
Plot a 3D operating envelope:
- x-axis: distance (m)
- y-axis: goodput (bps, log10 display)
- z-axis: energy/bit (uJ, log10 display)
- color: RSSI (dBm)

Highlights the non-dominated goodput/energy points within each tested distance.
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

from plot_config import FIGSIZE_IEEE_DOUBLE, IEEE_FONTSIZE, SAVE_DPI, save_plot_outputs
from plot_rssi_vs_multiple import (
    _add_rotated_rssi_scale,
    _position_3d_z_label_top,
    _style_3d_axes,
    collect_rssi_data,
    setup_plot_style,
)


ANNOTATED_DISTANCES = (6.25, 25.0, 50.0, 75.0, 100.0)
ANNOTATION_OFFSETS = {
    6.25: (1.4, 0.030, 0.050),
    25.0: (1.5, 0.040, -0.050),
    50.0: (1.5, 0.040, 0.045),
    75.0: (1.4, -0.035, 0.040),
    100.0: (-3.0, 0.040, 0.040),
}
ANNOTATION_HA = {
    100.0: "right",
}
DISTANCE_TICKS_M = np.asarray([6.25, 25.0, 50.0, 75.0, 100.0], dtype=float)
GOODPUT_TICKS_BPS = np.asarray([10.0, 30.0, 100.0, 300.0, 1000.0, 1500.0], dtype=float)
ENERGY_TICKS_UJ = np.asarray([20.0, 50.0, 100.0, 300.0, 1000.0, 2500.0], dtype=float)
FRONTIER_LINE_COLOR = (0.20, 0.20, 0.20, 0.78)


def _is_dataset_root(data_root):
    root_name = os.path.basename(os.path.normpath(data_root)).lower()
    return "dataset" in root_name


def _default_output_paths(data_root):
    is_dataset = _is_dataset_root(data_root)
    plot_root = "dataset_plots" if is_dataset else "raw_test_data_plots"
    prefix = "dataset" if is_dataset else "raw"
    out_dir = os.path.join(WORKSPACE, "results", plot_root, "operating_space_3d")
    output_png = os.path.join(out_dir, f"{prefix}_distance_goodput_energy_3d.png")
    frontier_csv = os.path.join(out_dir, f"{prefix}_distance_goodput_energy_frontier.csv")
    return output_png, frontier_csv


def _output_base(output_path):
    base, ext = os.path.splitext(output_path)
    return base if ext else output_path


def _fmt_distance(distance):
    return f"{distance:.2f}".rstrip("0").rstrip(".")


def _build_operating_points(records):
    points = []
    for distance, sf, bw, tp, metrics in records:
        throughput = metrics.get("throughput_bps")
        energy_per_bit = metrics.get("energy_per_bit_uj")
        per_pct = metrics.get("per_pct")
        rssi_avg = metrics.get("rssi_avg")
        values = (throughput, energy_per_bit, per_pct, rssi_avg)
        if not all(value is not None and np.isfinite(value) for value in values):
            continue
        if throughput <= 0.0 or energy_per_bit <= 0.0:
            continue
        goodput = float(throughput) * max(0.0, 1.0 - float(per_pct) / 100.0)
        points.append(
            {
                "distance_m": float(distance),
                "sf": int(sf),
                "bw_hz": int(bw),
                "tp_dbm": int(tp),
                "throughput_bps": float(throughput),
                "goodput_bps": float(goodput),
                "energy_per_bit_uj": float(energy_per_bit),
                "per_pct": float(per_pct),
                "rssi_avg": float(rssi_avg),
            }
        )
    return points


def _objective_vector(point):
    return np.array(
        [
            point["energy_per_bit_uj"],
            -point["goodput_bps"],
        ],
        dtype=float,
    )


def _pareto_front(points):
    frontier = []
    objectives = [_objective_vector(point) for point in points]
    for idx, objective in enumerate(objectives):
        dominated = False
        for other_idx, other_objective in enumerate(objectives):
            if other_idx == idx:
                continue
            if np.all(other_objective <= objective) and np.any(other_objective < objective):
                dominated = True
                break
        if not dominated:
            frontier.append(points[idx])
    return frontier


def _frontier_sort_key(point):
    return (
        point["goodput_bps"],
        point["energy_per_bit_uj"],
        point["per_pct"],
        point["sf"],
        point["bw_hz"],
        point["tp_dbm"],
    )


def _annotation_sort_key(point):
    return (
        point["goodput_bps"] / max(point["energy_per_bit_uj"], 1e-12),
        point["goodput_bps"],
        -point["energy_per_bit_uj"],
        -point["per_pct"],
    )


def _compute_frontier_by_distance(points):
    grouped = defaultdict(list)
    for point in points:
        grouped[point["distance_m"]].append(point)

    frontier_by_distance = {}
    for distance in sorted(grouped):
        frontier = _pareto_front(grouped[distance])
        frontier.sort(key=_frontier_sort_key)
        frontier_by_distance[distance] = frontier
    return frontier_by_distance


def _write_frontier_csv(frontier_by_distance, output_csv):
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "distance_m",
                "sf",
                "bw_hz",
                "tp_dbm",
                "goodput_bps",
                "throughput_bps",
                "energy_per_bit_uj",
                "per_pct",
                "rssi_avg",
                "frontier_index_within_distance",
            ],
        )
        writer.writeheader()
        for distance in sorted(frontier_by_distance):
            for idx, point in enumerate(frontier_by_distance[distance]):
                writer.writerow(
                    {
                        "distance_m": point["distance_m"],
                        "sf": point["sf"],
                        "bw_hz": point["bw_hz"],
                        "tp_dbm": point["tp_dbm"],
                        "goodput_bps": point["goodput_bps"],
                        "throughput_bps": point["throughput_bps"],
                        "energy_per_bit_uj": point["energy_per_bit_uj"],
                        "per_pct": point["per_pct"],
                        "rssi_avg": point["rssi_avg"],
                        "frontier_index_within_distance": idx,
                    }
                )


def _distance_label(distance):
    return f"{_fmt_distance(distance)} m"


def _annotation_point(frontier_points):
    return max(frontier_points, key=_annotation_sort_key)


def _plot_points(ax, points, *, cmap, rssi_min, rssi_max, alpha, size, linewidths, edgecolors):
    x_vals = np.asarray([point["distance_m"] for point in points], dtype=float)
    y_vals = np.log10(np.asarray([point["goodput_bps"] for point in points], dtype=float))
    z_vals = np.log10(np.asarray([point["energy_per_bit_uj"] for point in points], dtype=float))
    rssi_vals = np.asarray([point["rssi_avg"] for point in points], dtype=float)
    return ax.scatter(
        x_vals,
        y_vals,
        z_vals,
        c=rssi_vals,
        cmap=cmap,
        vmin=rssi_min,
        vmax=rssi_max,
        s=size,
        alpha=alpha,
        edgecolors=edgecolors,
        linewidths=linewidths,
        depthshade=False,
    )


def plot_operating_space_3d(points, frontier_by_distance, output_png):
    if not points:
        raise RuntimeError("No valid operating points found for the operating-space plot.")

    cmap = plt.get_cmap("viridis")
    rssi_vals = np.asarray([point["rssi_avg"] for point in points], dtype=float)
    rssi_min = float(np.min(rssi_vals))
    rssi_max = float(np.max(rssi_vals))
    if abs(rssi_max - rssi_min) < 1e-9:
        rssi_max = rssi_min + 1.0

    fig = plt.figure(figsize=FIGSIZE_IEEE_DOUBLE)
    ax = fig.add_subplot(111, projection="3d")

    base_scatter = _plot_points(
        ax,
        points,
        cmap=cmap,
        rssi_min=rssi_min,
        rssi_max=rssi_max,
        alpha=0.22,
        size=18,
        linewidths=0.0,
        edgecolors="none",
    )

    frontier_points = []
    for distance in sorted(frontier_by_distance):
        frontier = frontier_by_distance[distance]
        if not frontier:
            continue
        frontier_points.extend(frontier)
        if len(frontier) < 2:
            continue
        x_vals = np.full(len(frontier), float(distance), dtype=float)
        y_vals = np.log10(np.asarray([point["goodput_bps"] for point in frontier], dtype=float))
        z_vals = np.log10(np.asarray([point["energy_per_bit_uj"] for point in frontier], dtype=float))
        ax.plot(
            x_vals,
            y_vals,
            z_vals,
            color=FRONTIER_LINE_COLOR,
            linewidth=1.1,
            alpha=0.92,
            zorder=4,
        )

    if frontier_points:
        _plot_points(
            ax,
            frontier_points,
            cmap=cmap,
            rssi_min=rssi_min,
            rssi_max=rssi_max,
            alpha=0.98,
            size=46,
            linewidths=0.35,
            edgecolors="black",
        )

    ax.set_xlabel("Distance (m)", fontsize=IEEE_FONTSIZE, labelpad=1.8)
    ax.set_ylabel("Goodput (bps)", fontsize=IEEE_FONTSIZE, labelpad=2.4)
    _position_3d_z_label_top(ax, r"Energy/bit ($\mu$J)", 1.03, 0.84, fontsize=IEEE_FONTSIZE)

    ax.set_xticks(DISTANCE_TICKS_M)
    ax.set_xticklabels([_fmt_distance(distance) for distance in DISTANCE_TICKS_M])
    ax.set_yticks(np.log10(GOODPUT_TICKS_BPS))
    ax.set_yticklabels([f"{tick:.0f}" for tick in GOODPUT_TICKS_BPS])
    ax.set_zticks(np.log10(ENERGY_TICKS_UJ))
    ax.set_zticklabels([f"{tick:.0f}" for tick in ENERGY_TICKS_UJ])

    ax.set_xlim(5.0, 102.0)
    ax.set_ylim(np.log10(8.0), np.log10(1700.0))
    ax.set_zlim(np.log10(15.0), np.log10(2800.0))

    _style_3d_axes(ax, elev=24, azim=-58, box_aspect=(1.25, 1.0, 0.92))
    ax.tick_params(axis="x", labelsize=IEEE_FONTSIZE, pad=-0.2)
    ax.tick_params(axis="y", labelsize=IEEE_FONTSIZE, pad=-0.8)
    ax.tick_params(axis="z", labelsize=IEEE_FONTSIZE, pad=-0.2)

    for distance in ANNOTATED_DISTANCES:
        frontier = frontier_by_distance.get(distance)
        if not frontier:
            continue
        point = _annotation_point(frontier)
        dx, dy, dz = ANNOTATION_OFFSETS[distance]
        ax.text(
            point["distance_m"] + dx,
            np.log10(point["goodput_bps"]) + dy,
            np.log10(point["energy_per_bit_uj"]) + dz,
            _distance_label(distance),
            fontsize=IEEE_FONTSIZE,
            ha=ANNOTATION_HA.get(distance, "left"),
            va="bottom",
            color="black",
        )

    fig.subplots_adjust(left=0.03, right=0.95, top=0.93, bottom=0.06)
    _add_rotated_rssi_scale(
        fig,
        ax,
        base_scatter.cmap,
        float(base_scatter.norm.vmin),
        float(base_scatter.norm.vmax),
        label=r"RSSI (dBm)",
        fontsize=IEEE_FONTSIZE,
        scale_graph_side="right",
        label_graph_side="left",
    )
    png_path, pdf_path = save_plot_outputs(fig, output_png, dpi=SAVE_DPI, bbox_inches="tight", save_pdf=True)
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot a 3D distance-goodput-energy operating envelope colored by RSSI."
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join(WORKSPACE, "raw_test_data"),
        help="Dataset root to read (raw_test_data or dataset).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path for the operating-space plot.",
    )
    parser.add_argument(
        "--frontier-csv",
        default=None,
        help="Optional CSV path for the per-distance goodput/energy frontier points.",
    )
    args = parser.parse_args()

    root_name = os.path.basename(os.path.normpath(args.data_root)).lower()
    if "smoothed" in root_name:
        raise ValueError("dataset_smoothed is not supported for this plot; use raw_test_data/ or dataset/.")

    default_output_png, default_frontier_csv = _default_output_paths(args.data_root)
    if args.output is None:
        args.output = default_output_png
    if args.frontier_csv is None:
        if args.output == default_output_png:
            args.frontier_csv = default_frontier_csv
        else:
            args.frontier_csv = _output_base(args.output) + "_frontier.csv"

    setup_plot_style()
    points = _build_operating_points(collect_rssi_data(args.data_root))
    if not points:
        raise RuntimeError("No valid operating points found.")

    frontier_by_distance = _compute_frontier_by_distance(points)
    _write_frontier_csv(frontier_by_distance, args.frontier_csv)
    png_path, pdf_path = plot_operating_space_3d(points, frontier_by_distance, args.output)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {args.frontier_csv}")


if __name__ == "__main__":
    main()
