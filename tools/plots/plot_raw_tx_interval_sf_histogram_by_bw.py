"""
3D TX-interval histogram:
- x-axis: SF
- y-axis: BW
- z-axis: TX interval variation
- bar color: configurable extra metric aggregated over files
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
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
from plot_config import FIGSIZE_ONE_COL, IEEE_FONTSIZE, SAVE_DPI, save_plot_outputs
from plot_rssi_vs_multiple import (
    _add_rotated_rssi_scale,
    _fmt_bw,
    _position_3d_z_label_top,
    _style_3d_axes,
)


SF_VALUES = [7, 8, 9, 10, 11, 12]
BW_VALUES = [62500, 125000, 250000, 500000]
TP_VALUES = [2, 12, 22]
BW_VALUES_DESC = sorted(BW_VALUES, reverse=True)
AIRTIME_FILENAME = "airtime_by_sf_bw_payload.csv"

CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")
DIST_RE = re.compile(r"^distance_([\d.]+)m?$")

COLOR_METRICS = {
    "throughput": ("mean_throughput_bps", "Throughput (bps)", "viridis"),
    "energy_packet": ("mean_energy_packet_mj", "Energy/packet (mJ)", "magma"),
    "energy_bit": ("mean_energy_bit_uj", r"Energy/bit ($\mu$J)", "magma"),
    "airtime": ("mean_airtime_ms", "Mean airtime (ms)", "magma"),
    "gap": ("mean_quiet_gap_ms", "Mean off-air gap (ms)", "viridis"),
    "std_tx": ("std_tx_interval_ms", "Std. dev. of TX interval (ms)", "plasma"),
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


def _payload_size_col(header):
    return header.index("payload_size_bytes") if "payload_size_bytes" in header else None


def _resolve_airtime_table_path(data_root):
    candidates = [
        os.path.join(data_root, AIRTIME_FILENAME),
        os.path.join(WORKSPACE, AIRTIME_FILENAME),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Could not find {AIRTIME_FILENAME} next to {data_root} or in {WORKSPACE}.")


def load_airtime_table(data_root):
    airtime_by_payload = {}
    airtime_by_cfg = defaultdict(list)
    with open(_resolve_airtime_table_path(data_root), "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                bw_hz = int(row["bw"])
                payload_size = int(row["payload_size"])
                sf = int(row["sf"])
                airtime_ms = float(row["airtime_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            airtime_by_payload[(sf, bw_hz, payload_size)] = airtime_ms
            airtime_by_cfg[(sf, bw_hz)].append(airtime_ms)
    return airtime_by_payload, {key: float(np.mean(values)) for key, values in airtime_by_cfg.items()}


def _energy_columns(header):
    if "energy_per_packet_min_mj" in header and "energy_per_packet_max_mj" in header:
        return header.index("energy_per_packet_min_mj"), header.index("energy_per_packet_max_mj"), 1.0
    if "energy_per_packet_j_min" in header and "energy_per_packet_j_max" in header:
        return header.index("energy_per_packet_j_min"), header.index("energy_per_packet_j_max"), 1000.0
    return None, None, 1.0


def _read_file_tx_airtime_stats(path, sf, bw_hz, airtime_by_payload, fallback_airtime_ms):
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return None

    header = rows[0]
    tx_idx = _tx_interval_col(header)
    if tx_idx is None:
        return None
    payload_size_idx = _payload_size_col(header)
    payload_idx = header.index("payload") if "payload" in header else None
    energy_min_idx, energy_max_idx, energy_scale = _energy_columns(header)

    tx_values = []
    airtime_values = []
    payload_sizes = []
    energy_packet_values = []
    for row in rows[1:]:
        if len(row) <= tx_idx:
            continue
        if payload_idx is not None and len(row) > payload_idx:
            payload = row[payload_idx].strip()
            if payload.startswith("CFG "):
                continue
        try:
            if row[tx_idx]:
                tx_values.append(float(row[tx_idx]))
        except Exception:
            pass
        if payload_size_idx is None or len(row) <= payload_size_idx:
            continue
        try:
            payload_size = int(float(row[payload_size_idx])) if row[payload_size_idx] else None
        except Exception:
            payload_size = None
        if payload_size is None or payload_size <= 0:
            continue
        payload_sizes.append(float(payload_size))
        airtime_ms = airtime_by_payload.get((sf, bw_hz, payload_size))
        if airtime_ms is not None:
            airtime_values.append(float(airtime_ms))
        if energy_min_idx is None or energy_max_idx is None or len(row) <= max(energy_min_idx, energy_max_idx):
            continue
        try:
            energy_min = float(row[energy_min_idx])
            energy_max = float(row[energy_max_idx])
        except Exception:
            continue
        energy_packet_values.append(0.5 * (energy_min + energy_max) * energy_scale)

    if not tx_values:
        return None

    mean_tx_interval_ms = float(np.mean(tx_values))
    mean_airtime_ms = float(np.mean(airtime_values)) if airtime_values else float(fallback_airtime_ms)
    quiet_gap_ms = max(mean_tx_interval_ms - mean_airtime_ms, 0.0)
    mean_payload_bytes = float(np.mean(payload_sizes)) if payload_sizes else None
    throughput_bps = None
    if mean_payload_bytes is not None and mean_tx_interval_ms > 0:
        throughput_bps = (mean_payload_bytes * 8.0) / (mean_tx_interval_ms / 1000.0)
    mean_energy_packet_mj = float(np.mean(energy_packet_values)) if energy_packet_values else None
    mean_energy_bit_uj = None
    if mean_energy_packet_mj is not None and mean_payload_bytes is not None and mean_payload_bytes > 0:
        mean_energy_bit_uj = mean_energy_packet_mj * 1000.0 / (mean_payload_bytes * 8.0)

    return {
        "mean_tx_interval_ms": mean_tx_interval_ms,
        "std_tx_interval_ms": float(np.std(tx_values)) if len(tx_values) > 1 else 0.0,
        "max_tx_interval_ms": float(np.max(tx_values)),
        "mean_airtime_ms": mean_airtime_ms,
        "mean_quiet_gap_ms": quiet_gap_ms,
        "mean_payload_bytes": mean_payload_bytes,
        "mean_throughput_bps": throughput_bps,
        "mean_energy_packet_mj": mean_energy_packet_mj,
        "mean_energy_bit_uj": mean_energy_bit_uj,
    }


def collect_tx_interval_file_stats(data_root, airtime_by_payload, fallback_airtime_by_cfg):
    """Collect per-file TX interval and airtime-derived stats for each SF/BW/TP config."""
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
                fallback_airtime_ms = fallback_airtime_by_cfg.get((sf, bw))
                if fallback_airtime_ms is None:
                    continue
                stats = _read_file_tx_airtime_stats(
                    file_path,
                    sf,
                    bw,
                    airtime_by_payload,
                    fallback_airtime_ms,
                )
                if stats is None:
                    continue
                records.append(
                    {
                        "distance_m": distance,
                        "sf": sf,
                        "bw_hz": bw,
                        "tp_dbm": tp,
                        **stats,
                    }
                )
    return records


def aggregate_tx_interval_by_sf_bw(file_records):
    """Aggregate file-level metrics across tested distances and TP for each SF/BW pair."""
    grouped = defaultdict(list)
    for record in file_records:
        grouped[(record["sf"], record["bw_hz"])].append(record)

    rows = []
    for sf in SF_VALUES:
        for bw in BW_VALUES_DESC:
            bucket = grouped.get((sf, bw), [])
            if not bucket:
                continue
            tx_means = np.asarray([row["mean_tx_interval_ms"] for row in bucket], dtype=float)
            tx_std_values = np.asarray([row["std_tx_interval_ms"] for row in bucket], dtype=float)
            airtime_values = np.asarray([row["mean_airtime_ms"] for row in bucket], dtype=float)
            gap_values = np.asarray([row["mean_quiet_gap_ms"] for row in bucket], dtype=float)
            throughput_values = np.asarray(
                [row["mean_throughput_bps"] for row in bucket if row.get("mean_throughput_bps") is not None],
                dtype=float,
            )
            energy_packet_values = np.asarray(
                [row["mean_energy_packet_mj"] for row in bucket if row.get("mean_energy_packet_mj") is not None],
                dtype=float,
            )
            energy_bit_values = np.asarray(
                [row["mean_energy_bit_uj"] for row in bucket if row.get("mean_energy_bit_uj") is not None],
                dtype=float,
            )
            rows.append(
                {
                    "sf": sf,
                    "bw_hz": bw,
                    "mean_tx_interval_ms": float(np.mean(tx_means)),
                    "std_tx_interval_ms": float(np.std(tx_means)) if tx_means.size > 1 else 0.0,
                    "mean_std_tx_interval_ms": float(np.mean(tx_std_values)),
                    "std_std_tx_interval_ms": float(np.std(tx_std_values)) if tx_std_values.size > 1 else 0.0,
                    "mean_airtime_ms": float(np.mean(airtime_values)),
                    "mean_quiet_gap_ms": float(np.mean(gap_values)),
                    "mean_throughput_bps": float(np.mean(throughput_values)) if throughput_values.size else None,
                    "mean_energy_packet_mj": float(np.mean(energy_packet_values)) if energy_packet_values.size else None,
                    "mean_energy_bit_uj": float(np.mean(energy_bit_values)) if energy_bit_values.size else None,
                    "file_count": float(len(bucket)),
                }
            )
    return rows


def _norm_from_rows(rows, key):
    values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=float)
    if values.size == 0:
        raise RuntimeError(f"No valid values found for '{key}'.")
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _is_dataset_root(data_root):
    root_name = os.path.basename(os.path.normpath(data_root)).lower()
    return "dataset" in root_name


def _default_tx_output_paths(data_root):
    is_dataset = _is_dataset_root(data_root)
    plot_root = "dataset_plots" if is_dataset else "raw_test_data_plots"
    prefix = "dataset" if is_dataset else "raw"
    out_dir = os.path.join(WORKSPACE, "results", plot_root, "tx_interval")
    return (
        os.path.join(out_dir, f"{prefix}_tx_interval_variation_sf_histogram_by_bw.png"),
        os.path.join(out_dir, f"{prefix}_tx_interval_variation_sf_ribbons_by_bw.png"),
    )


def _normalize_png_path(output_path):
    return output_path if output_path.lower().endswith(".png") else f"{output_path}.png"


def _set_tx_axes(ax, z_max, z_label):
    ax.set_xlabel("SF", labelpad=1.6)
    ax.set_ylabel("BW (kHz)", labelpad=1.8)
    _position_3d_z_label_top(ax, z_label, 1.03, 0.86, fontsize=IEEE_FONTSIZE)
    ax.set_xticks(np.arange(len(SF_VALUES), dtype=float))
    ax.set_xticklabels([str(sf) for sf in SF_VALUES])
    ax.set_yticks(np.arange(len(BW_VALUES_DESC), dtype=float))
    ax.set_yticklabels([_fmt_bw(bw / 1000.0) for bw in BW_VALUES_DESC])
    ax.set_xlim(-0.6, len(SF_VALUES) - 0.4)
    ax.set_ylim(-0.6, len(BW_VALUES_DESC) - 0.4)
    ax.set_zlim(0.0, z_max * 1.10 if z_max > 0 else 1.0)


def _add_tx_stand_scale(fig, ax, cmap, norm, color_label):
    _add_rotated_rssi_scale(
        fig,
        ax,
        cmap,
        float(norm.vmin),
        float(norm.vmax),
        label=color_label,
        fontsize=IEEE_FONTSIZE,
        label_graph_side="left",
        scale_graph_side="stand",
        stand_values_side="left",
    )


def plot_tx_interval_3d_histogram(agg_rows, output_png, color_by="throughput"):
    if not agg_rows:
        return

    color_key, color_label, cmap_name = COLOR_METRICS[color_by]
    norm = _norm_from_rows(agg_rows, color_key)
    cmap = plt.get_cmap(cmap_name)
    row_lookup = {(row["sf"], row["bw_hz"]): row for row in agg_rows}

    fig = plt.figure(figsize=FIGSIZE_ONE_COL)
    ax = fig.add_subplot(111, projection="3d")

    dx = 0.64
    dy = 0.64
    z_max = 0.0
    for sf_idx, sf in enumerate(SF_VALUES):
        for bw_idx, bw in enumerate(BW_VALUES_DESC):
            row = row_lookup.get((sf, bw))
            if row is None:
                continue
            z_height = float(row["mean_std_tx_interval_ms"])
            if not np.isfinite(z_height):
                continue
            z_max = max(z_max, z_height)
            color_raw = row.get(color_key)
            if color_raw is None or not np.isfinite(float(color_raw)):
                continue
            color_value = float(color_raw)
            ax.bar3d(
                sf_idx - dx / 2.0,
                bw_idx - dy / 2.0,
                0.0,
                dx,
                dy,
                z_height,
                color=cmap(norm(color_value)),
                edgecolor="black",
                linewidth=0.35,
                shade=True,
                alpha=0.98,
            )

    _set_tx_axes(ax, z_max, "TX interval std. dev. (ms)")
    _style_3d_axes(ax, elev=26, azim=-56, box_aspect=(1.08, 1.0, 0.8))
    _add_tx_stand_scale(
        fig,
        ax,
        cmap,
        norm,
        color_label,
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08)
    png_path, pdf_path = save_plot_outputs(fig, output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def _iter_contiguous_segments(points):
    if not points:
        return
    segment = [points[0]]
    for point in points[1:]:
        if point[0] == segment[-1][0] + 1:
            segment.append(point)
        else:
            if len(segment) >= 2:
                yield segment
            segment = [point]
    if len(segment) >= 2:
        yield segment


def plot_tx_interval_3d_ribbons(agg_rows, output_png, color_by="throughput"):
    if not agg_rows:
        return

    color_key, color_label, cmap_name = COLOR_METRICS[color_by]
    norm = _norm_from_rows(agg_rows, color_key)
    cmap = plt.get_cmap(cmap_name)
    row_lookup = {(row["sf"], row["bw_hz"]): row for row in agg_rows}

    fig = plt.figure(figsize=FIGSIZE_ONE_COL)
    ax = fig.add_subplot(111, projection="3d")

    z_max = 0.0
    for bw_idx, bw in enumerate(BW_VALUES_DESC):
        row_points = []
        for sf_idx, sf in enumerate(SF_VALUES):
            row = row_lookup.get((sf, bw))
            if row is None:
                continue
            z_height = row.get("mean_std_tx_interval_ms")
            color_raw = row.get(color_key)
            if z_height is None or color_raw is None:
                continue
            z_height = float(z_height)
            color_value = float(color_raw)
            if not (np.isfinite(z_height) and np.isfinite(color_value)):
                continue
            z_max = max(z_max, z_height)
            row_points.append((sf_idx, z_height, color_value))

        for segment in _iter_contiguous_segments(row_points):
            x_vals = np.asarray([point[0] for point in segment], dtype=float)
            y_vals = np.full(x_vals.shape, float(bw_idx))
            z_vals = np.asarray([point[1] for point in segment], dtype=float)
            color_vals = np.asarray([point[2] for point in segment], dtype=float)

            segment_polys = []
            segment_colors = []
            for idx in range(len(segment) - 1):
                x0, z0, c0 = segment[idx]
                x1, z1, c1 = segment[idx + 1]
                segment_polys.append([(x0, 0.0), (x0, z0), (x1, z1), (x1, 0.0)])
                segment_colors.append(cmap(norm(0.5 * (c0 + c1))))

            poly = PolyCollection(
                segment_polys,
                facecolors=segment_colors,
                edgecolors="none",
                alpha=0.55,
            )
            ax.add_collection3d(poly, zs=[float(bw_idx)] * len(segment_polys), zdir="y")
            ax.plot(
                x_vals,
                y_vals,
                z_vals,
                color="black",
                linewidth=1.3,
                alpha=0.95,
                zorder=6,
            )
            ax.scatter(
                x_vals,
                y_vals,
                z_vals,
                c=color_vals,
                cmap=cmap,
                vmin=float(norm.vmin),
                vmax=float(norm.vmax),
                s=28,
                edgecolors="black",
                linewidths=0.25,
                depthshade=False,
                zorder=7,
            )

    _set_tx_axes(ax, z_max, "TX interval std. dev. (ms)")
    _style_3d_axes(ax, elev=26, azim=-56, box_aspect=(1.08, 1.0, 0.8))
    _add_tx_stand_scale(
        fig,
        ax,
        cmap,
        norm,
        color_label,
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08)
    png_path, pdf_path = save_plot_outputs(fig, output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot a 3D TX-interval variation histogram over SF and BW.")
    parser.add_argument(
        "--data-root",
        default=os.path.join(WORKSPACE, "raw_test_data"),
        help="Dataset root (default: raw_test_data/).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path for the bar plot (default: results/.../tx_interval/).",
    )
    parser.add_argument(
        "--ribbon-output",
        default=None,
        help="Output PNG path for the filled 3D line plot (default: derived from the bar-plot output path).",
    )
    parser.add_argument(
        "--color-by",
        choices=sorted(COLOR_METRICS.keys()),
        default="throughput",
        help="Metric used to color the bars.",
    )
    args = parser.parse_args()

    default_bar_output, default_ribbon_output = _default_tx_output_paths(args.data_root)
    if args.output is None:
        args.output = default_bar_output
    if args.ribbon_output is None:
        if args.output == default_bar_output:
            args.ribbon_output = default_ribbon_output
        else:
            output_png = _normalize_png_path(args.output)
            args.ribbon_output = f"{os.path.splitext(output_png)[0]}_ribbons.png"

    setup_plot_style()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(os.path.dirname(args.ribbon_output), exist_ok=True)
    airtime_by_payload, fallback_airtime_by_cfg = load_airtime_table(args.data_root)
    file_records = collect_tx_interval_file_stats(args.data_root, airtime_by_payload, fallback_airtime_by_cfg)
    if not file_records:
        raise RuntimeError("No TX-interval data found.")
    agg_rows = aggregate_tx_interval_by_sf_bw(file_records)
    plot_tx_interval_3d_histogram(agg_rows, args.output, color_by=args.color_by)
    plot_tx_interval_3d_ribbons(agg_rows, args.ribbon_output, color_by=args.color_by)


if __name__ == "__main__":
    main()
