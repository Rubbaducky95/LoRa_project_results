"""
Plot RSSI (avg, raw) vs various x-axis metrics.
Generates multiple plots to show impact on RSSI from distance, SF, BW, TP, energy, throughput, etc.
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
from matplotlib.cm import ScalarMappable
from matplotlib import transforms as mtransforms
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
from plot_config import FIGSIZE_IEEE_DOUBLE, FIGSIZE_ONE_COL, FIGSIZE_TWO_COL, IEEE_FONTSIZE, SAVE_DPI

CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")
DIST_RE = re.compile(r"^distance_([\d.]+)m?$")

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
            "axes.labelsize": IEEE_FONTSIZE,
            "xtick.labelsize": IEEE_FONTSIZE,
            "ytick.labelsize": IEEE_FONTSIZE,
        }
    )


def parse_cfg(filename):
    m = CFG_RE.match(filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def parse_distance(folder_name):
    m = DIST_RE.match(folder_name)
    return float(m.group(1)) if m else None


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None


def read_file_metrics(path):
    """
    Return dict with trusted per-file metrics derived from packet rows only.
    Returns None if insufficient data.
    """
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows:
        return None
    header = rows[0]
    if "rssi" not in header:
        return None
    i_rssi = header.index("rssi")
    i_emin = header.index("energy_per_packet_min_mj") if "energy_per_packet_min_mj" in header else None
    i_emax = header.index("energy_per_packet_max_mj") if "energy_per_packet_max_mj" in header else None
    if i_emin is None and "energy_per_packet_j_min" in header:
        i_emin = header.index("energy_per_packet_j_min")
        i_emax = header.index("energy_per_packet_j_max")
        energy_scale = 1000.0
    else:
        energy_scale = 1.0
    i_time = header.index("time_since_transmission_init_ms") if "time_since_transmission_init_ms" in header else None
    i_payload = header.index("payload_size_bytes") if "payload_size_bytes" in header else None
    i_payload_raw = header.index("payload") if "payload" in header else None

    rssi_vals = []
    energy_mids = []
    times_ms = []
    payload_bytes = []
    total_packets = 0
    lost_packets = 0
    for row in rows[1:]:
        if len(row) <= i_rssi:
            continue
        payload_raw = row[i_payload_raw].strip() if i_payload_raw is not None and len(row) > i_payload_raw else ""
        if payload_raw.startswith("CFG "):
            continue
        if payload_raw:
            total_packets += 1
            if payload_raw == "PACKET_LOST":
                lost_packets += 1
        r = parse_float(row[i_rssi])
        if r is not None:
            rssi_vals.append(r)
        if i_emin is not None and i_emax is not None and len(row) > max(i_emin, i_emax):
            e1, e2 = parse_float(row[i_emin]), parse_float(row[i_emax])
            if e1 is not None and e2 is not None:
                energy_mids.append((e1 + e2) / 2.0 * energy_scale)
        if i_time is not None and len(row) > i_time:
            t = parse_float(row[i_time])
            if t is not None:
                times_ms.append(t)
        if i_payload is not None and len(row) > i_payload:
            p = parse_float(row[i_payload])
            if p is not None and p > 0:
                payload_bytes.append(int(p))

    if not rssi_vals:
        return None

    if energy_scale == 1.0 and energy_mids:
        sample = energy_mids[: min(20, len(energy_mids))]
        if sum(sample) / len(sample) < 1.0:
            energy_mids = [e * 1000.0 for e in energy_mids]

    rssi_avg = sum(rssi_vals) / len(rssi_vals)
    rssi_std = float(np.std(rssi_vals)) if len(rssi_vals) >= 2 else 0.0
    energy_mj = sum(energy_mids) / len(energy_mids) if energy_mids else None
    count = total_packets or len(rssi_vals)
    duration_s = (max(times_ms) - min(times_ms)) / 1000.0 if len(times_ms) >= 2 else 0.0
    payload_b = int(sum(payload_bytes) / len(payload_bytes)) if payload_bytes else 37
    total_bits = count * payload_b * 8
    throughput_bps = total_bits / duration_s if duration_s > 0 else None
    energy_per_bit_uj = (sum(energy_mids) * 1000.0 / total_bits) if energy_mids and total_bits > 0 else None
    per_pct = (100.0 * lost_packets / count) if count > 0 else None

    return {
        "rssi_avg": rssi_avg,
        "rssi_std": rssi_std,
        "energy_mj": energy_mj,
        "throughput_bps": throughput_bps,
        "energy_per_bit_uj": energy_per_bit_uj,
        "per_pct": per_pct,
        "count": count,
    }


def collect_rssi_data(data_root):
    """
    Per file: (distance, sf, bw, tp) -> metrics dict.
    Returns list of (distance, sf, bw, tp, metrics).
    """
    records = []
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
                if sf not in SF_VALUES or bw not in BW_VALUES or tp not in TP_VALUES:
                    continue
                metrics = read_file_metrics(os.path.join(root, fn))
                if metrics is None:
                    continue
                records.append((distance, sf, bw, tp, metrics))
    return records


SF_BW_METRIC_KEYS = (
    "rssi_avg",
    "rssi_std",
    "energy_mj",
    "throughput_bps",
    "energy_per_bit_uj",
    "per_pct",
    "count",
)


def _get_sf_bw_order():
    return [(sf, bw) for sf in SF_VALUES for bw in BW_VALUES]


def _sf_bw_label(sf, bw):
    return f"{sf},{_fmt_bw(bw / 1000.0)}"


def aggregate_rssi_by_distance_sf_bw(records):
    """Aggregate per-file metrics over TP for each (distance, SF, BW) tuple."""
    grouped = defaultdict(lambda: defaultdict(list))
    for distance, sf, bw, tp, metrics in records:
        bucket = grouped[(distance, sf, bw)]
        bucket["tp_values"].append(tp)
        for metric_key in SF_BW_METRIC_KEYS:
            value = metrics.get(metric_key)
            if value is not None:
                bucket[metric_key].append(float(value))

    aggregated = []
    for (distance, sf, bw), bucket in sorted(grouped.items()):
        row = {"distance": distance, "sf": sf, "bw": bw, "tp_count": len(bucket["tp_values"])}
        for metric_key in SF_BW_METRIC_KEYS:
            values = bucket.get(metric_key, [])
            row[metric_key] = float(np.mean(values)) if values else None
        aggregated.append(row)
    return aggregated


def _build_distance_cfg_matrix(agg_records, metric_key):
    cfgs = _get_sf_bw_order()
    distances = sorted({row["distance"] for row in agg_records})
    cfg_to_idx = {cfg: i for i, cfg in enumerate(cfgs)}
    dist_to_idx = {distance: i for i, distance in enumerate(distances)}
    mat = np.full((len(cfgs), len(distances)), np.nan)
    for row in agg_records:
        value = row.get(metric_key)
        if value is None:
            continue
        cfg_idx = cfg_to_idx[(row["sf"], row["bw"])]
        dist_idx = dist_to_idx[row["distance"]]
        mat[cfg_idx, dist_idx] = float(value)
    return np.array(distances, dtype=float), cfgs, mat


def _build_distance_sf_matrix(agg_records, bw, metric_key):
    distances = sorted({row["distance"] for row in agg_records if row["bw"] == bw})
    dist_to_idx = {distance: i for i, distance in enumerate(distances)}
    sf_to_idx = {sf: i for i, sf in enumerate(SF_VALUES)}
    mat = np.full((len(SF_VALUES), len(distances)), np.nan)
    for row in agg_records:
        if row["bw"] != bw:
            continue
        value = row.get(metric_key)
        if value is None:
            continue
        mat[sf_to_idx[row["sf"]], dist_to_idx[row["distance"]]] = float(value)
    return np.array(distances, dtype=float), np.array(SF_VALUES, dtype=float), mat


def _fmt_distance(d):
    return str(int(d)) if abs(d - round(d)) < 1e-6 else f"{d:.2f}".rstrip("0").rstrip(".")


def _set_distance_ticks(ax, distances):
    tick_values = distances[::2] if len(distances) > 8 else distances
    if tick_values[-1] != distances[-1]:
        tick_values = np.append(tick_values, distances[-1])
    ax.set_xticks(tick_values)
    ax.set_xticklabels([_fmt_distance(d) for d in tick_values])


def _set_cfg_ticks(ax, cfgs, step=2):
    idxs = list(range(0, len(cfgs), step))
    if idxs[-1] != len(cfgs) - 1:
        idxs.append(len(cfgs) - 1)
    ax.set_yticks(idxs)
    ax.set_yticklabels([_sf_bw_label(*cfgs[i]) for i in idxs])


def _style_3d_axes(ax, elev=27, azim=-61, box_aspect=(1.55, 1.0, 0.82)):
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(axis="x", labelsize=IEEE_FONTSIZE - 1, pad=1)
    ax.tick_params(axis="y", labelsize=IEEE_FONTSIZE - 1, pad=1)
    ax.tick_params(axis="z", labelsize=IEEE_FONTSIZE - 1, pad=1)
    try:
        ax.set_box_aspect(box_aspect)
    except Exception:
        pass
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.10)
        pane.set_edgecolor((0.35, 0.35, 0.35, 0.25))
    ax.grid(True, alpha=0.35)


def _norm_from_matrix(mat):
    values = np.asarray(mat, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _plot_rssi_colored_surface(
    ax,
    x_values,
    y_values,
    z_matrix,
    color_matrix,
    color_norm,
    z_label,
    y_label,
    y_tick_labels=None,
    y_tick_step=1,
    panel_text=None,
    color_cmap="viridis",
    project_floor=False,
):
    if np.all(np.isnan(z_matrix)) or np.all(np.isnan(color_matrix)):
        return False

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    z_matrix = np.asarray(z_matrix, dtype=float)
    color_matrix = np.asarray(color_matrix, dtype=float)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    z_masked = np.ma.masked_invalid(z_matrix)
    color_fill = np.where(np.isnan(color_matrix), color_norm.vmin, color_matrix)
    cmap_obj = plt.get_cmap(color_cmap)
    facecolors = cmap_obj(color_norm(color_fill))
    z_min = float(np.nanmin(z_matrix))
    z_max = float(np.nanmax(z_matrix))
    z_span = max(z_max - z_min, 1.0)
    z_floor = z_min - (0.18 if project_floor else 0.08) * z_span

    ax.plot_surface(
        x_grid,
        y_grid,
        z_masked,
        facecolors=facecolors,
        rcount=z_matrix.shape[0],
        ccount=z_matrix.shape[1],
        edgecolor=(0.08, 0.08, 0.08, 0.18),
        linewidth=0.35,
        antialiased=True,
        shade=False,
        alpha=0.98,
    )
    if project_floor:
        ax.contourf(
            x_grid,
            y_grid,
            np.ma.masked_invalid(color_matrix),
            zdir="z",
            offset=z_floor,
            levels=np.linspace(color_norm.vmin, color_norm.vmax, 9),
            cmap=color_cmap,
            norm=color_norm,
            alpha=0.90,
        )

    ax.set_xlim(float(x_values.min()), float(x_values.max()))
    ax.set_ylim(float(y_values.min()), float(y_values.max()))
    ax.set_zlim(z_floor, z_max + 0.06 * z_span)
    ax.set_xlabel("Distance (m)", fontsize=IEEE_FONTSIZE, labelpad=3)
    ax.set_ylabel(y_label, fontsize=IEEE_FONTSIZE, labelpad=5)
    ax.set_zlabel(z_label, fontsize=IEEE_FONTSIZE, labelpad=4)
    ax.zaxis.label.set_verticalalignment("top")
    _set_distance_ticks(ax, x_values)
    if y_tick_labels is None:
        ax.set_yticks(y_values[::y_tick_step])
        ax.set_yticklabels([str(int(v)) if abs(v - round(v)) < 1e-6 else str(v) for v in y_values[::y_tick_step]])
    else:
        idxs = list(range(0, len(y_values), y_tick_step))
        if idxs[-1] != len(y_values) - 1:
            idxs.append(len(y_values) - 1)
        ax.set_yticks(y_values[idxs])
        ax.set_yticklabels([y_tick_labels[i] for i in idxs])
    _style_3d_axes(ax)
    if panel_text:
        ax.text2D(
            0.03,
            0.95,
            panel_text,
            transform=ax.transAxes,
            fontsize=IEEE_FONTSIZE,
            va="top",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.55", alpha=0.85),
        )
    return True


def _add_metric_colorbar(fig, axes, norm, label, cmap="viridis"):
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.80, pad=0.02, aspect=26)
    cbar.set_label(label, fontsize=IEEE_FONTSIZE)
    cbar.ax.tick_params(labelsize=IEEE_FONTSIZE - 1)


def plot_rssi_distance_sfbw_surface(agg_records, output_png):
    """Surface: x=distance, y=SF/BW config, z=avg RSSI, color=energy per bit."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    distances, cfgs, rssi_mat = _build_distance_cfg_matrix(agg_records, "rssi_avg")
    if np.all(np.isnan(rssi_mat)):
        return
    _, _, energy_bit_mat = _build_distance_cfg_matrix(agg_records, "energy_per_bit_uj")
    color_mat = energy_bit_mat if not np.all(np.isnan(energy_bit_mat)) else rssi_mat
    color_norm = _norm_from_matrix(color_mat)
    color_label = r"Energy per bit ($\mu$J)" if not np.all(np.isnan(energy_bit_mat)) else r"RSSI (avg) (dBm)"
    color_cmap = "cividis" if not np.all(np.isnan(energy_bit_mat)) else "viridis"
    fig = plt.figure(figsize=FIGSIZE_IEEE_DOUBLE)
    ax = fig.add_subplot(111, projection="3d")
    _plot_rssi_colored_surface(
        ax,
        distances,
        np.arange(len(cfgs), dtype=float),
        rssi_mat,
        color_mat,
        color_norm,
        r"RSSI (avg) (dBm)",
        "SF/BW cfg",
        y_tick_labels=[_sf_bw_label(*cfg) for cfg in cfgs],
        y_tick_step=2,
        color_cmap=color_cmap,
        project_floor=False,
    )
    _add_metric_colorbar(fig, ax, color_norm, color_label, cmap=color_cmap)
    fig.subplots_adjust(left=0.04, right=0.90, top=0.96, bottom=0.06)
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


def plot_rssi_distance_sfbw_bars(agg_records, output_png):
    """3D bars: x=distance, y=SF/BW config, z=avg RSSI, color=PER."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    distances, cfgs, rssi_mat = _build_distance_cfg_matrix(agg_records, "rssi_avg")
    if np.all(np.isnan(rssi_mat)):
        return
    _, _, per_mat = _build_distance_cfg_matrix(agg_records, "per_pct")
    rssi_min = float(np.nanmin(rssi_mat))
    rssi_max = float(np.nanmax(rssi_mat))
    rssi_span = max(rssi_max - rssi_min, 1.0)
    rssi_floor = rssi_min - 0.22 * rssi_span
    per_norm = _norm_from_matrix(per_mat) if not np.all(np.isnan(per_mat)) else None

    x_grid, y_grid = np.meshgrid(distances, np.arange(len(cfgs), dtype=float))
    mask = ~np.isnan(rssi_mat)
    if not np.any(mask):
        return
    dist_step = float(np.min(np.diff(distances))) if len(distances) > 1 else 1.0
    dx = 0.68 * dist_step
    dy = 0.58

    fig = plt.figure(figsize=FIGSIZE_IEEE_DOUBLE)
    ax = fig.add_subplot(111, projection="3d")
    if per_norm is not None:
        bar_colors = plt.cm.inferno(per_norm(np.where(mask, per_mat, per_norm.vmin)[mask]))
    else:
        bar_colors = np.tile(np.array([[0.28, 0.52, 0.64, 0.95]]), (mask.sum(), 1))
    ax.bar3d(
        x_grid[mask] - dx / 2.0,
        y_grid[mask] - dy / 2.0,
        np.full(mask.sum(), rssi_floor),
        np.full(mask.sum(), dx),
        np.full(mask.sum(), dy),
        rssi_mat[mask] - rssi_floor,
        color=bar_colors,
        edgecolor=(0.08, 0.08, 0.08, 0.9),
        linewidth=0.18,
        shade=True,
        alpha=0.97,
    )
    ax.set_xlim(float(distances.min()), float(distances.max()))
    ax.set_ylim(0, len(cfgs) - 1)
    ax.set_zlim(rssi_floor, rssi_max + 0.04 * rssi_span)
    ax.set_xlabel("Distance (m)", fontsize=IEEE_FONTSIZE, labelpad=3)
    ax.set_ylabel("SF/BW cfg", fontsize=IEEE_FONTSIZE, labelpad=5)
    ax.set_zlabel(r"RSSI (avg) (dBm)", fontsize=IEEE_FONTSIZE, labelpad=4)
    ax.zaxis.label.set_verticalalignment("top")
    _set_distance_ticks(ax, distances)
    _set_cfg_ticks(ax, cfgs, step=2)
    _style_3d_axes(ax, elev=25, azim=-63)
    if per_norm is not None:
        _add_metric_colorbar(fig, ax, per_norm, "PER (%)", cmap="inferno")
    fig.subplots_adjust(left=0.04, right=0.90, top=0.96, bottom=0.06)
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


def plot_rssi_distance_sf_by_bw_surfaces(agg_records, output_png):
    """2x2 BW panels: x=distance, y=SF, z=avg RSSI, color=energy per bit."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    mats = []
    for bw in BW_VALUES:
        distances, sf_vals, rssi_mat = _build_distance_sf_matrix(agg_records, bw, "rssi_avg")
        _, _, energy_bit_mat = _build_distance_sf_matrix(agg_records, bw, "energy_per_bit_uj")
        mats.append((bw, distances, sf_vals, rssi_mat, energy_bit_mat))
    valid_rssi_mats = [rssi_mat for _, _, _, rssi_mat, _ in mats if not np.all(np.isnan(rssi_mat))]
    valid_energy_mats = [energy_bit_mat for _, _, _, _, energy_bit_mat in mats if not np.all(np.isnan(energy_bit_mat))]
    if not valid_rssi_mats:
        return
    color_cmap = "cividis" if valid_energy_mats else "viridis"
    color_label = r"Energy per bit ($\mu$J)" if valid_energy_mats else r"RSSI (avg) (dBm)"
    color_norm = _norm_from_matrix(
        np.vstack(valid_energy_mats) if valid_energy_mats else np.vstack(valid_rssi_mats)
    )

    fig = plt.figure(figsize=FIGSIZE_IEEE_DOUBLE)
    axes = []
    for idx, (bw, distances, sf_vals, rssi_mat, energy_bit_mat) in enumerate(mats):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        axes.append(ax)
        if np.all(np.isnan(rssi_mat)):
            ax.set_axis_off()
            continue
        color_mat = energy_bit_mat if not np.all(np.isnan(energy_bit_mat)) else rssi_mat
        _plot_rssi_colored_surface(
            ax,
            distances,
            sf_vals,
            rssi_mat,
            color_mat,
            color_norm,
            r"RSSI (avg) (dBm)",
            "SF",
            y_tick_labels=[str(sf) for sf in SF_VALUES],
            y_tick_step=1,
            panel_text=f"BW {_fmt_bw(bw / 1000.0)} kHz",
            color_cmap=color_cmap,
            project_floor=False,
        )
    _add_metric_colorbar(fig, axes, color_norm, color_label, cmap=color_cmap)
    fig.subplots_adjust(left=0.02, right=0.92, top=0.96, bottom=0.05, wspace=0.02, hspace=0.08)
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


def plot_rssi_distance_signal_tradeoff_surfaces(agg_records, output_png):
    """Two-panel tradeoff view: energy/PER as z, RSSI as the color cue."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    distances, cfgs, rssi_mat = _build_distance_cfg_matrix(agg_records, "rssi_avg")
    if np.all(np.isnan(rssi_mat)):
        return
    metric_specs = [
        ("energy_per_bit_uj", r"Energy per bit ($\mu$J)"),
        ("per_pct", "PER (%)"),
    ]
    panel_mats = []
    for metric_key, metric_label in metric_specs:
        _, _, metric_mat = _build_distance_cfg_matrix(agg_records, metric_key)
        if not np.all(np.isnan(metric_mat)):
            panel_mats.append((metric_label, metric_mat))
    if not panel_mats:
        return

    rssi_norm = _norm_from_matrix(rssi_mat)
    fig = plt.figure(figsize=FIGSIZE_IEEE_DOUBLE)
    axes = []
    for idx, (metric_label, metric_mat) in enumerate(panel_mats):
        ax = fig.add_subplot(1, len(panel_mats), idx + 1, projection="3d")
        axes.append(ax)
        _plot_rssi_colored_surface(
            ax,
            distances,
            np.arange(len(cfgs), dtype=float),
            metric_mat,
            rssi_mat,
            rssi_norm,
            metric_label,
            "SF/BW cfg",
            y_tick_labels=[_sf_bw_label(*cfg) for cfg in cfgs],
            y_tick_step=3,
            color_cmap="viridis",
            project_floor=False,
        )
    _add_metric_colorbar(fig, axes, rssi_norm, r"RSSI (avg) (dBm)", cmap="viridis")
    fig.subplots_adjust(left=0.03, right=0.92, top=0.96, bottom=0.06, wspace=0.08)
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


def plot_rssi_vs_x(records, x_key, x_label, output_png, x_unit=""):
    """Scatter: RSSI (avg) vs x_key. Color by SF."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    points = []
    for distance, sf, bw, tp, m in records:
        x_val = None
        if x_key == "distance":
            x_val = distance
        elif x_key == "sf":
            x_val = sf
        elif x_key == "bw":
            x_val = bw / 1000.0
        elif x_key == "tp":
            x_val = tp
        elif x_key == "energy_mj":
            x_val = m.get("energy_mj")
        elif x_key == "throughput_bps":
            x_val = m.get("throughput_bps")
        elif x_key == "energy_per_bit_uj":
            x_val = m.get("energy_per_bit_uj")
        if x_val is None:
            continue
        points.append((sf, bw, x_val, m["rssi_avg"]))

    if not points:
        return

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_ONE_COL)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(SF_VALUES)))
    for sf_idx, sf in enumerate(SF_VALUES):
        pts = [p for p in points if p[0] == sf]
        if not pts:
            continue
        x_vals = [p[2] for p in pts]
        y_vals = [p[3] for p in pts]
        ax.scatter(x_vals, y_vals, c=[colors[sf_idx]], label=f"SF{sf}", s=25, marker="o", edgecolors="k", linewidths=0.3)
    ax.set_xlabel(x_label + (f" ({x_unit})" if x_unit else ""), fontsize=IEEE_FONTSIZE)
    ax.set_ylabel(r"RSSI (avg) (dBm)", fontsize=IEEE_FONTSIZE)
    ax.legend(loc="best", fontsize=IEEE_FONTSIZE, ncol=2)
    ax.set_xlim(left=min(p[2] for p in points) * 0.98 if min(p[2] for p in points) > 0 else None)
    fig.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.12)
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


def _fmt_bw(b):
    return f"{b:.1f}".rstrip("0").rstrip(".") if b != int(b) else str(int(b))


def _get_axis_val(axis, distance, sf, bw, tp):
    if axis == "distance":
        return distance
    if axis == "sf":
        return sf
    if axis == "bw":
        return bw / 1000.0
    if axis == "tp":
        return tp
    return None


def _nice_rssi_ticks(vmin, vmax):
    """Pick simple 10 dBm ticks for the inset RSSI scale."""
    lo = int(np.ceil(vmin / 10.0) * 10)
    hi = int(np.floor(vmax / 10.0) * 10)
    if hi <= lo:
        return [float(vmin), 0.5 * (vmin + vmax), float(vmax)]
    if hi == lo + 10:
        return [lo, lo + 5, hi]
    return [lo, 0.5 * (lo + hi), hi]


def _add_rotated_rssi_scale(fig, ax, cmap, vmin, vmax, label, angle_deg=45, loc=(0.02, 0.72), size=(0.34, 0.23), fontsize=None, text_angle_deg=None):
    """Add a compact rotated gradient scale in the upper-left of a 3D axes."""
    if fontsize is None:
        fontsize = IEEE_FONTSIZE
    if text_angle_deg is None:
        text_angle_deg = angle_deg
    ax_pos = ax.get_position()
    scale_ax = fig.add_axes(
        [
            ax_pos.x0 + loc[0] * ax_pos.width,
            ax_pos.y0 + loc[1] * ax_pos.height,
            size[0] * ax_pos.width,
            size[1] * ax_pos.height,
        ],
        zorder=20,
    )
    scale_ax.set_xlim(0, 1)
    scale_ax.set_ylim(0, 1)
    scale_ax.axis("off")

    grad_x0, grad_y0 = 0.08, 0.18
    grad_w, grad_h = 1.2, 0.12
    tick_len = 0.07
    tick_gap = 0.045
    label_gap = 0.16
    cx = grad_x0 + 0.4 * grad_w
    cy = grad_y0 + 0.8 * grad_h
    rot = mtransforms.Affine2D().rotate_deg_around(cx, cy, angle_deg) + scale_ax.transAxes

    gradient = np.linspace(vmin, vmax, 256).reshape(1, -1)
    scale_ax.imshow(
        gradient,
        extent=(grad_x0, grad_x0 + grad_w, grad_y0, grad_y0 + grad_h),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=rot,
        interpolation="bilinear",
        clip_on=False,
        zorder=2,
    )
    scale_ax.add_patch(
        Rectangle(
            (grad_x0, grad_y0),
            grad_w,
            grad_h,
            fill=False,
            edgecolor="black",
            linewidth=0.8,
            transform=rot,
            clip_on=False,
            zorder=3,
        )
    )

    for tick in _nice_rssi_ticks(vmin, vmax):
        frac = 0.5 if vmax == vmin else (tick - vmin) / (vmax - vmin)
        x_tick = grad_x0 + frac * grad_w
        scale_ax.plot(
            [x_tick, x_tick],
            [grad_y0 + grad_h, grad_y0 + grad_h + tick_len],
            color="black",
            linewidth=0.8,
            transform=rot,
            clip_on=False,
            zorder=4,
        )
        tick_text = f"{tick:.0f}" if abs(tick - round(tick)) < 1e-6 else f"{tick:.1f}"
        scale_ax.text(
            x_tick,
            grad_y0 + grad_h + tick_len + tick_gap,
            tick_text,
            transform=rot,
            rotation=text_angle_deg,
            rotation_mode="anchor",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            clip_on=False,
            zorder=5,
        )

    scale_ax.text(
        grad_x0 + 0.5 * grad_w,
        grad_y0 + grad_h + tick_len + label_gap,
        label,
        transform=rot,
        rotation=text_angle_deg,
        rotation_mode="anchor",
        ha="center",
        va="top",
        fontsize=fontsize,
        clip_on=False,
        zorder=5,
    )
    return scale_ax


def plot_rssi_3d(records, output_png, x_axis, y_axis, z_axis, invert_x=False, invert_y=False):
    """3D scatter: x_axis, y_axis, z_axis as axes, color=RSSI (avg). Aggregates over the 4th dimension."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    agg = defaultdict(list)
    for distance, sf, bw, tp, m in records:
        key = (_get_axis_val(x_axis, distance, sf, bw, tp),
               _get_axis_val(y_axis, distance, sf, bw, tp),
               _get_axis_val(z_axis, distance, sf, bw, tp))
        if None in key:
            continue
        agg[key].append(m["rssi_avg"])
    points = []
    for (xv, yv, zv), rssi_list in agg.items():
        points.append({x_axis: xv, y_axis: yv, z_axis: zv, "rssi": sum(rssi_list) / len(rssi_list)})
    if not points:
        return
    x_vals = np.array([p[x_axis] for p in points])
    y_vals = np.array([p[y_axis] for p in points])
    z_vals = np.array([p[z_axis] for p in points])
    rssi_vals = np.array([p["rssi"] for p in points])
    fig = plt.figure(figsize=FIGSIZE_IEEE_DOUBLE)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x_vals, y_vals, z_vals, c=rssi_vals, cmap="viridis", s=50, edgecolors="k", linewidths=0.3)
    labels = {"sf": "SF", "bw": "BW (kHz)", "tp": "TP (dBm)", "distance": "Distance (m)"}
    ax.set_xlabel(labels.get(x_axis, x_axis), fontsize=IEEE_FONTSIZE, labelpad=2)
    ax.set_ylabel(labels.get(y_axis, y_axis), fontsize=IEEE_FONTSIZE, labelpad=2)
    ax.set_zlabel(labels.get(z_axis, z_axis), fontsize=IEEE_FONTSIZE, labelpad=2)
    ax.zaxis.label.set_verticalalignment('top')
    ax.tick_params(axis='x', pad=1)
    ax.tick_params(axis='y', pad=1)
    ax.tick_params(axis='z', pad=1)
    bw_khz = sorted(set(bw / 1000.0 for bw in BW_VALUES))
    distances = sorted(set(d for d, _, _, _, _ in records))
    for axis, key in [("x", x_axis), ("y", y_axis), ("z", z_axis)]:
        set_ticks = getattr(ax, f"set_{axis}ticks")
        set_ticklabels = getattr(ax, f"set_{axis}ticklabels")
        if key == "sf":
            set_ticks(SF_VALUES)
            set_ticklabels([str(s) for s in SF_VALUES])
        elif key == "bw":
            set_ticks(bw_khz)
            set_ticklabels([_fmt_bw(b) for b in bw_khz])
        elif key == "tp":
            set_ticks(TP_VALUES)
            set_ticklabels([str(t) for t in TP_VALUES])
        elif key == "distance":
            set_ticks(distances)
            set_ticklabels([str(int(d)) if d == int(d) else f"{d:.1f}" for d in distances])
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05)
    _add_rotated_rssi_scale(
        fig,
        ax,
        sc.cmap,
        float(sc.norm.vmin),
        float(sc.norm.vmax),
        r"RSSI (avg) (dBm)",
        angle_deg=225,
        text_angle_deg=45,
        loc=(0.00, 0.71),
        size=(0.38, 0.28),
    )
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


DISTANCE_TICK_INTERVAL = 25.0


def _config_label(sf, bw, tp):
    """Short label for config: SF7 BW62.5 kHz TP2 -> S7/B62.5/T2."""
    bw_khz = bw / 1000.0
    s = f"{bw_khz:.1f}".rstrip("0").rstrip(".")
    return f"S{sf}/B{s}/T{tp}"


def _get_config_order():
    """Yield (sf, bw, tp) in consistent order: SF, BW, TP."""
    for sf in SF_VALUES:
        for bw in BW_VALUES:
            for tp in TP_VALUES:
                yield (sf, bw, tp)


def plot_rssi_config_vs_distance_heatmap(records, output_png):
    """
    Heatmap: x-axis = distance, y-axis = config (SF/BW/TP), color = avg RSSI.
    RSSI is the simple per-file average from read_file_metrics.
    """
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    configs = list(_get_config_order())
    distances = sorted(set(d for d, _, _, _, _ in records))
    cfg_to_idx = {c: i for i, c in enumerate(configs)}
    mat = np.full((len(distances), len(configs)), np.nan)
    for distance, sf, bw, tp, m in records:
        cfg = (sf, bw, tp)
        if cfg not in cfg_to_idx:
            continue
        d_idx = distances.index(distance)
        c_idx = cfg_to_idx[cfg]
        mat[d_idx, c_idx] = m["rssi_avg"]
    mat = mat.T
    if np.all(np.isnan(mat)):
        return
    rssi_min = np.nanmin(mat)
    rssi_max = np.nanmax(mat)
    d_arr = np.array(distances)
    if len(distances) == 1:
        x_edges = np.array([d_arr[0] - 1, d_arr[0], d_arr[0] + 1])
    else:
        half = np.diff(d_arr) / 2
        midpoints = (d_arr[:-1] + d_arr[1:]) / 2
        x_edges = np.concatenate(([d_arr[0] - half[0]], midpoints, [d_arr[-1] + half[-1]]))
    y_edges = np.arange(len(configs) + 1)
    fig, ax = plt.subplots(figsize=FIGSIZE_IEEE_DOUBLE)
    im = ax.pcolormesh(x_edges, y_edges, mat, cmap="viridis", vmin=rssi_min, vmax=rssi_max, shading="flat")
    ax.set_xlim(min(distances), max(distances))
    ax.set_ylim(0, len(configs))
    ax.set_xlabel("Distance (m)", fontsize=IEEE_FONTSIZE)
    ax.set_ylabel("Config (SF/BW kHz/TP)", fontsize=IEEE_FONTSIZE)
    ax.set_yticks(np.arange(len(configs)) + 0.5)
    ax.set_yticklabels([_config_label(sf, bw, tp) for sf, bw, tp in configs])
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=25)
    cbar.set_label(r"RSSI (avg) (dBm)", fontsize=IEEE_FONTSIZE)
    cbar.ax.tick_params(labelsize=IEEE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=IEEE_FONTSIZE)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.18)
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


def plot_rssi_config_distance_energy_3d(records, output_png):
    """
    3D scatter: x=distance, y=config, z=energy (mJ), color=RSSI (avg).
    Each point is one file with simple per-file avg RSSI.
    """
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    configs = list(_get_config_order())
    cfg_to_idx = {c: i for i, c in enumerate(configs)}
    points = []
    for distance, sf, bw, tp, m in records:
        cfg = (sf, bw, tp)
        if cfg not in cfg_to_idx:
            continue
        energy = m.get("energy_mj")
        if energy is None or energy <= 0:
            continue
        points.append((distance, cfg_to_idx[cfg], energy, m["rssi_avg"]))
    if not points:
        return
    x_vals = np.array([p[0] for p in points])
    y_vals = np.array([p[1] for p in points])
    z_vals = np.array([p[2] for p in points])
    rssi_vals = np.array([p[3] for p in points])
    rssi_min, rssi_max = rssi_vals.min(), rssi_vals.max()
    fig = plt.figure(figsize=FIGSIZE_IEEE_DOUBLE)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x_vals, y_vals, z_vals, c=rssi_vals, cmap="viridis", s=25, edgecolors="k", linewidths=0.2,
                    vmin=rssi_min, vmax=rssi_max)
    ax.set_xlabel("Distance (m)", fontsize=IEEE_FONTSIZE, labelpad=2)
    ax.set_ylabel("Config", fontsize=IEEE_FONTSIZE, labelpad=2)
    ax.set_zlabel("Energy (mJ)", fontsize=IEEE_FONTSIZE, labelpad=2)
    ax.zaxis.label.set_verticalalignment("top")
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", pad=1)
    ax.tick_params(axis="z", pad=1)
    ax.set_yticks(np.arange(len(configs)))
    ax.set_yticklabels([_config_label(sf, bw, tp) for sf, bw, tp in configs])
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=20)
    cbar.set_label(r"RSSI (avg) (dBm)", fontsize=IEEE_FONTSIZE)
    cbar.ax.tick_params(labelsize=IEEE_FONTSIZE)
    fig.subplots_adjust(left=0.02, right=0.92, top=0.95, bottom=0.05)
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


def plot_rssi_3d_combined(records, output_png):
    """Three 3D subplots in one 2-column figure: A) BW-distance-SF, B) TP-distance-SF, C) TP-distance-BW."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    distances = sorted(set(d for d, _, _, _, _ in records))
    d_min, d_max = min(distances), max(distances)
    dist_ticks = [DISTANCE_TICK_INTERVAL * i for i in range(1, int(d_max / DISTANCE_TICK_INTERVAL) + 1) if d_min <= DISTANCE_TICK_INTERVAL * i <= d_max]
    bw_khz = sorted(set(bw / 1000.0 for bw in BW_VALUES))
    labels = {"sf": "SF", "bw": "BW (kHz)", "tp": "TP (dBm)", "distance": "Distance (m)"}

    def agg_3d(x_axis, y_axis, z_axis):
        a = defaultdict(list)
        for distance, sf, bw, tp, m in records:
            key = (_get_axis_val(x_axis, distance, sf, bw, tp),
                   _get_axis_val(y_axis, distance, sf, bw, tp),
                   _get_axis_val(z_axis, distance, sf, bw, tp))
            if None in key:
                continue
            a[key].append(m["rssi_avg"])
        return [{x_axis: xv, y_axis: yv, z_axis: zv, "rssi": sum(v) / len(v)} for (xv, yv, zv), v in a.items()]

    configs = [
        ("bw", "distance", "sf", True, False),
        ("tp", "distance", "sf", True, False),
        ("tp", "distance", "bw", True, False),
    ]
    fig = plt.figure(figsize=FIGSIZE_IEEE_DOUBLE)
    rssi_all = []
    for x_axis, y_axis, z_axis, _, _ in configs:
        pts = agg_3d(x_axis, y_axis, z_axis)
        rssi_all.extend(p["rssi"] for p in pts)
    rssi_min, rssi_max = min(rssi_all), max(rssi_all) if rssi_all else (-100, -40)

    scatter_handles = []
    for idx, (x_axis, y_axis, z_axis, inv_x, inv_y) in enumerate(configs):
        pts = agg_3d(x_axis, y_axis, z_axis)
        if not pts:
            continue
        x_vals = np.array([p[x_axis] for p in pts])
        y_vals = np.array([p[y_axis] for p in pts])
        z_vals = np.array([p[z_axis] for p in pts])
        if x_axis == "bw" or y_axis == "bw" or z_axis == "bw":
            for axis, key in [("x", x_axis), ("y", y_axis), ("z", z_axis)]:
                if key == "bw":
                    if axis == "x":
                        x_vals = np.log10(np.maximum(x_vals, 1))
                    elif axis == "y":
                        y_vals = np.log10(np.maximum(y_vals, 1))
                    else:
                        z_vals = np.log10(np.maximum(z_vals, 1))
                    break
        rssi_vals = np.array([p["rssi"] for p in pts])
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        sc = ax.scatter(x_vals, y_vals, z_vals, c=rssi_vals, cmap="viridis", s=35, edgecolors="k", linewidths=0.3, vmin=rssi_min, vmax=rssi_max)
        scatter_handles.append((sc, ax, x_axis, y_axis, z_axis))
        ax.set_xlabel(labels[x_axis], fontsize=IEEE_FONTSIZE, labelpad=2)
        ax.set_ylabel(labels[y_axis], fontsize=IEEE_FONTSIZE, labelpad=2)
        ax.set_zlabel(labels[z_axis], fontsize=IEEE_FONTSIZE, labelpad=2)
        ax.zaxis.label.set_verticalalignment('top')
        ax.tick_params(axis='x', pad=1)
        ax.tick_params(axis='y', pad=1)
        ax.tick_params(axis='z', pad=1)
        for axis, key in [("x", x_axis), ("y", y_axis), ("z", z_axis)]:
            set_ticks = getattr(ax, f"set_{axis}ticks")
            set_ticklabels = getattr(ax, f"set_{axis}ticklabels")
            if key == "sf":
                set_ticks(SF_VALUES)
                set_ticklabels([str(s) for s in SF_VALUES])
            elif key == "bw":
                log_bw = np.log10(np.array(bw_khz))
                set_ticks(log_bw)
                set_ticklabels([_fmt_bw(b) for b in bw_khz])
            elif key == "tp":
                set_ticks(TP_VALUES)
                set_ticklabels([str(t) for t in TP_VALUES])
            elif key == "distance":
                set_ticks(dist_ticks)
                set_ticklabels([f"{int(d)}" for d in dist_ticks])
        if inv_x:
            ax.invert_xaxis()
        if inv_y:
            ax.invert_yaxis()

    fig.subplots_adjust(left=0.02, right=0.95, top=0.92, bottom=0.05, wspace=0.2)
    for sc, ax, _, _, _ in scatter_handles:
        _add_rotated_rssi_scale(
            fig,
            ax,
            sc.cmap,
            float(sc.norm.vmin),
            float(sc.norm.vmax),
            r"RSSI (avg) (dBm)",
            angle_deg=225,
            text_angle_deg=40,
            loc=(0.00, 0.75),
            size=(0.42, 0.28),
            fontsize=IEEE_FONTSIZE,
        )
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Plot RSSI (avg) vs various x-axis metrics.")
    parser.add_argument(
        "--data-root",
        default=os.path.join(WORKSPACE, "raw_test_data"),
        help="Dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for PNGs.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(WORKSPACE, "results", "raw_test_data_plots", "rssi")

    setup_plot_style()
    records = collect_rssi_data(args.data_root)
    if not records:
        raise RuntimeError("No RSSI data found.")
    agg_records = aggregate_rssi_by_distance_sf_bw(records)

    plot_rssi_3d_combined(records, os.path.join(args.output_dir, "raw_rssi_3d_combined.png"))
    plot_rssi_distance_sfbw_surface(agg_records, os.path.join(args.output_dir, "raw_rssi_distance_sfbw_surface.png"))
    plot_rssi_distance_sfbw_bars(agg_records, os.path.join(args.output_dir, "raw_rssi_distance_sfbw_bars.png"))
    plot_rssi_distance_sf_by_bw_surfaces(
        agg_records,
        os.path.join(args.output_dir, "raw_rssi_distance_sf_by_bw_surfaces.png"),
    )
    plot_rssi_distance_signal_tradeoff_surfaces(
        agg_records,
        os.path.join(args.output_dir, "raw_rssi_distance_signal_tradeoff.png"),
    )


if __name__ == "__main__":
    main()
