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
from mpl_toolkits.mplot3d import Axes3D, proj3d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
from plot_config import FIGSIZE_IEEE_DOUBLE, FIGSIZE_ONE_COL, FIGSIZE_TWO_COL, IEEE_FONTSIZE, SAVE_DPI, save_plot_outputs

CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")
DIST_RE = re.compile(r"^distance_([\d.]+)m?$")

SF_VALUES = [7, 8, 9, 10, 11, 12]
TP_VALUES = [2, 12, 22]
BW_VALUES = [62500, 125000, 250000, 500000]
STANDARD_DISTANCE_TICKS = np.array([25.0, 50.0, 75.0, 100.0], dtype=float)

DEBUG_DARK_3D_BACKGROUND = False
DEBUG_3D_FIG_COLOR = "#171b22"
DEBUG_3D_AXES_COLOR = "#222833"
DEBUG_3D_TEXT_COLOR = "#f2f5f9"
DEBUG_3D_EDGE_RGBA = (0.92, 0.95, 0.99, 0.95)
DEBUG_3D_GRID_RGBA = (0.92, 0.95, 0.99, 0.18)
DEBUG_3D_PANE_RGBA = (0.24, 0.28, 0.35, 0.78)

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


def _fmt_distance(d):
    return str(int(d)) if abs(d - round(d)) < 1e-6 else f"{d:.2f}".rstrip("0").rstrip(".")


def _preferred_distance_ticks(distances):
    distances = np.asarray(distances, dtype=float)
    if distances.size == 0:
        return distances
    d_min = float(np.min(distances))
    d_max = float(np.max(distances))
    preferred = STANDARD_DISTANCE_TICKS[(STANDARD_DISTANCE_TICKS >= d_min) & (STANDARD_DISTANCE_TICKS <= d_max)]
    if preferred.size:
        return preferred
    if abs(d_max - d_min) < 1e-9:
        return np.array([d_min], dtype=float)
    return np.array([d_min, d_max], dtype=float)


def _set_distance_ticks(ax, distances):
    tick_values = _preferred_distance_ticks(distances)
    ax.set_xticks(tick_values)
    ax.set_xticklabels([_fmt_distance(d) for d in tick_values])


def _set_cfg_ticks(ax, cfgs, step=2):
    idxs = list(range(0, len(cfgs), step))
    if idxs[-1] != len(cfgs) - 1:
        idxs.append(len(cfgs) - 1)
    ax.set_yticks(idxs)
    ax.set_yticklabels([_sf_bw_label(*cfgs[i]) for i in idxs])


def _set_3d_tick_pads(ax, pad=0.5, labelsize=IEEE_FONTSIZE):
    """Set per-axis 3D tick-label padding with a scalar, tuple, or mapping."""
    axes = ("x", "y", "z")
    if isinstance(pad, dict):
        pads = {axis: float(pad.get(axis, 0.5)) for axis in axes}
    elif isinstance(pad, (tuple, list)) and len(pad) == 3:
        pads = {axis: float(axis_pad) for axis, axis_pad in zip(axes, pad)}
    else:
        scalar = float(pad)
        pads = {axis: scalar for axis in axes}

    for axis in axes:
        ax.tick_params(axis=axis, labelsize=labelsize, pad=pads[axis])


def _style_3d_axes(ax, elev=27, azim=-61, box_aspect=(1.55, 1.0, 0.82)):
    ax.view_init(elev=elev, azim=azim)
    _set_3d_tick_pads(ax, 0.3)
    try:
        ax.set_box_aspect(box_aspect)
    except Exception:
        pass
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.10)
        pane.set_edgecolor((0.35, 0.35, 0.35, 0.25))
    ax.grid(True, alpha=0.35)
    _apply_dark_3d_debug_theme(ax)


def _apply_dark_3d_debug_theme(ax):
    """Temporary high-contrast theme to make 3D cube edges easier to align."""
    if not DEBUG_DARK_3D_BACKGROUND:
        return

    fig = ax.figure
    fig.patch.set_facecolor(DEBUG_3D_FIG_COLOR)
    ax.set_facecolor(DEBUG_3D_AXES_COLOR)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(DEBUG_3D_PANE_RGBA)
        pane.set_edgecolor(DEBUG_3D_EDGE_RGBA)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis._axinfo["grid"]["color"] = DEBUG_3D_GRID_RGBA
            axis._axinfo["axisline"]["color"] = DEBUG_3D_EDGE_RGBA
            axis._axinfo["tick"]["color"] = DEBUG_3D_EDGE_RGBA
        except Exception:
            pass

    ax.tick_params(axis="x", colors=DEBUG_3D_TEXT_COLOR)
    ax.tick_params(axis="y", colors=DEBUG_3D_TEXT_COLOR)
    ax.tick_params(axis="z", colors=DEBUG_3D_TEXT_COLOR)
    ax.xaxis.label.set_color(DEBUG_3D_TEXT_COLOR)
    ax.yaxis.label.set_color(DEBUG_3D_TEXT_COLOR)
    ax.zaxis.label.set_color(DEBUG_3D_TEXT_COLOR)

    custom_zlabel = getattr(ax, "_custom_zlabel_text", None)
    if custom_zlabel is not None:
        text_items = custom_zlabel if isinstance(custom_zlabel, (list, tuple)) else [custom_zlabel]
        for item in text_items:
            try:
                item.set_color(DEBUG_3D_TEXT_COLOR)
            except Exception:
                pass

    rotated_scale_label = getattr(ax, "_rotated_scale_label_text", None)
    if rotated_scale_label is not None:
        try:
            rotated_scale_label.set_color(DEBUG_3D_TEXT_COLOR)
        except Exception:
            pass


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


def _set_3d_axis_labels(ax, x_label, y_label, z_label, x_pad=1.4, y_pad=2.0, z_pad=1.8):
    ax.set_xlabel(x_label, fontsize=IEEE_FONTSIZE, labelpad=x_pad)
    ax.set_ylabel(y_label, fontsize=IEEE_FONTSIZE, labelpad=y_pad)
    ax.set_zlabel(z_label, fontsize=IEEE_FONTSIZE, labelpad=z_pad)
    try:
        ax.zaxis.set_rotate_label(False)
        ax.zaxis.label.set_rotation(90)
    except Exception:
        pass
    ax.zaxis.label.set_verticalalignment("top")


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
    view_elev=27,
    view_azim=-61,
    box_aspect=(1.55, 1.0, 0.82),
    z_label_coords=None,
    z_label_rotation=0,
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
    _set_3d_axis_labels(ax, "Distance (m)", y_label, z_label)
    if z_label_coords is None:
        _clear_custom_z_label(ax)
    else:
        _position_3d_z_label_top(
            ax,
            z_label,
            *z_label_coords,
            fontsize=IEEE_FONTSIZE,
            rotation_deg=z_label_rotation,
        )
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
    _style_3d_axes(ax, elev=view_elev, azim=view_azim, box_aspect=box_aspect)
    if panel_text:
        ax.text2D(
            0.04,
            0.94,
            panel_text,
            transform=ax.transAxes,
            fontsize=IEEE_FONTSIZE - 1,
            va="top",
            bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="0.55", alpha=0.82),
        )
    return True


def _add_centered_horizontal_colorbar(fig, norm, label, cmap="viridis", rect=(0.34, 0.88, 0.32, 0.028)):
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes(rect, zorder=20)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(label, fontsize=IEEE_FONTSIZE, labelpad=2)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.tick_params(labelsize=IEEE_FONTSIZE, pad=1)
    return cbar


def plot_rssi_distance_sfbw_surface(agg_records, output_png):
    """Surface: x=distance, y=SF/BW config, z=avg RSSI, color=energy per bit."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    distances, cfgs, rssi_mat = _build_distance_cfg_matrix(agg_records, "rssi_avg")
    if np.all(np.isnan(rssi_mat)):
        return
    _, _, energy_bit_mat = _build_distance_cfg_matrix(agg_records, "energy_per_bit_uj")
    color_mat = energy_bit_mat if not np.all(np.isnan(energy_bit_mat)) else rssi_mat
    color_norm = _norm_from_matrix(color_mat)
    color_label = r"Energy/bit ($\mu$J)" if not np.all(np.isnan(energy_bit_mat)) else r"RSSI (dBm)"
    color_cmap = "cividis" if not np.all(np.isnan(energy_bit_mat)) else "viridis"
    fig = plt.figure(figsize=FIGSIZE_ONE_COL)
    ax = fig.add_subplot(111, projection="3d")
    _plot_rssi_colored_surface(
        ax,
        distances,
        np.arange(len(cfgs), dtype=float),
        rssi_mat,
        color_mat,
        color_norm,
        r"RSSI (dBm)",
        "SF/BW cfg",
        y_tick_labels=[_sf_bw_label(*cfg) for cfg in cfgs],
        y_tick_step=3,
        color_cmap=color_cmap,
        project_floor=False,
        view_elev=28,
        view_azim=-60,
        box_aspect=(1.50, 0.92, 0.88),
    )
    _add_centered_horizontal_colorbar(fig, color_norm, color_label, cmap=color_cmap, rect=(0.35, 0.885, 0.30, 0.028))
    fig.subplots_adjust(left=0.03, right=0.99, top=0.84, bottom=0.05)
    png_path, pdf_path = save_plot_outputs(fig, output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_rssi_distance_signal_tradeoff_surfaces(agg_records, output_png):
    """Single-panel tradeoff view: prefer PER as z, with RSSI as the color cue."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    distances, cfgs, rssi_mat = _build_distance_cfg_matrix(agg_records, "rssi_avg")
    if np.all(np.isnan(rssi_mat)):
        return
    metric_spec = None
    metric_mat = None
    for candidate_spec in [
        ("per_pct", "PER (%)"),
        ("energy_per_bit_uj", r"Energy/bit ($\mu$J)"),
    ]:
        _, _, candidate_mat = _build_distance_cfg_matrix(agg_records, candidate_spec[0])
        if not np.all(np.isnan(candidate_mat)):
            metric_spec = candidate_spec
            metric_mat = candidate_mat
            break
    if metric_spec is None or metric_mat is None:
        return

    rssi_norm = _norm_from_matrix(rssi_mat)
    fig = plt.figure(figsize=FIGSIZE_ONE_COL)
    ax = fig.add_subplot(111, projection="3d")
    _plot_rssi_colored_surface(
        ax,
        distances,
        np.arange(len(cfgs), dtype=float),
        metric_mat,
        rssi_mat,
        rssi_norm,
        metric_spec[1],
        "SF/BW cfg",
        y_tick_labels=[_sf_bw_label(*cfg) for cfg in cfgs],
        y_tick_step=3,
        color_cmap="viridis",
        project_floor=False,
        view_elev=28,
        view_azim=-60,
        box_aspect=(1.35, 0.92, 0.92),
        z_label_coords=(1.02, 0.84),
    )
    _add_rotated_rssi_scale(
        fig,
        ax,
        plt.get_cmap("viridis"),
        float(rssi_norm.vmin),
        float(rssi_norm.vmax),
        label=r"RSSI (dBm)",
        fontsize=IEEE_FONTSIZE,
        scale_graph_side="right",
        label_graph_side="left",
    )
    fig.subplots_adjust(left=0.03, right=0.99, top=0.92, bottom=0.05)
    png_path, pdf_path = save_plot_outputs(fig, output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


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
    png_path, pdf_path = save_plot_outputs(fig, output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


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


def _project_3d_graph_bbox_axes(fig, ax):
    """Estimate the visible 3D cube bounds in axes coordinates."""
    try:
        fig.canvas.draw()
    except Exception:
        pass

    try:
        x_vals = ax.get_xlim3d()
        y_vals = ax.get_ylim3d()
        z_vals = ax.get_zlim3d()
    except Exception:
        return (0.0, 0.0, 1.0, 1.0)

    corners = [
        (x, y, z)
        for x in x_vals
        for y in y_vals
        for z in z_vals
    ]
    points_axes = []
    proj = ax.get_proj()
    for x_val, y_val, z_val in corners:
        try:
            x_proj, y_proj, _ = proj3d.proj_transform(x_val, y_val, z_val, proj)
            x_disp, y_disp = ax.transData.transform((x_proj, y_proj))
            x_axes, y_axes = ax.transAxes.inverted().transform((x_disp, y_disp))
        except Exception:
            continue
        if np.isfinite(x_axes) and np.isfinite(y_axes):
            points_axes.append((float(x_axes), float(y_axes)))

    if not points_axes:
        return (0.0, 0.0, 1.0, 1.0)

    points_axes = np.asarray(points_axes, dtype=float)
    return (
        float(np.min(points_axes[:, 0])),
        float(np.min(points_axes[:, 1])),
        float(np.max(points_axes[:, 0])),
        float(np.max(points_axes[:, 1])),
    )


def _project_3d_point_axes(fig, ax, x_val, y_val, z_val):
    """Project one 3D data point into the host axes' normalized coordinates."""
    try:
        fig.canvas.draw()
    except Exception:
        pass

    try:
        proj = ax.get_proj()
        x_proj, y_proj, _ = proj3d.proj_transform(x_val, y_val, z_val, proj)
        x_disp, y_disp = ax.transData.transform((x_proj, y_proj))
        x_axes, y_axes = ax.transAxes.inverted().transform((x_disp, y_disp))
    except Exception:
        return None

    if not (np.isfinite(x_axes) and np.isfinite(y_axes)):
        return None
    return np.array([float(x_axes), float(y_axes)], dtype=float)


def _normalize_2d(vec, fallback):
    """Return a unit-length 2D vector, or the fallback if degenerate."""
    vec = np.asarray(vec, dtype=float)
    length = float(np.linalg.norm(vec))
    if length < 1e-9 or not np.isfinite(length):
        return np.asarray(fallback, dtype=float)
    return vec / length


def _resolve_xy_scale(scale, name):
    """Normalize a scalar or `(x, y)` scale into a float pair."""
    if np.isscalar(scale):
        value = float(scale)
        return (value, value)
    try:
        scale_x, scale_y = scale
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number or a 2-tuple") from None
    return (float(scale_x), float(scale_y))


def _resolve_scale_size(size, graph_w, graph_h, ax, fig):
    """Resolve `(w, h)` and infer one dimension from the drawn graph aspect."""
    if size is None:
        return None
    try:
        size_w, size_h = size
    except (TypeError, ValueError):
        raise ValueError("size must be a 2-tuple") from None

    try:
        fig.canvas.draw()
    except Exception:
        pass

    ax_bbox = ax.get_window_extent()
    width_basis = max(graph_w * float(ax_bbox.width), 1e-9)
    height_basis = max(graph_h * float(ax_bbox.height), 1e-9)

    if size_w is None and size_h is None:
        raise ValueError("size cannot have both width and height set to None")
    if size_w is None:
        size_h = float(size_h)
        size_w = size_h * height_basis / width_basis
    elif size_h is None:
        size_w = float(size_w)
        size_h = size_w * width_basis / height_basis
    else:
        size_w = float(size_w)
        size_h = float(size_h)
    return (size_w, size_h)


def _add_rotated_rssi_scale(
    fig,
    ax,
    cmap,
    vmin,
    vmax,
    label,
    *,
    scale_graph_side="left",
    label_graph_side=None,
    angle_deg=None,
    loc=None,
    label_loc=None,
    size=None,
    size_scale=1.0,
    loc_scale=1.0,
    fontsize=None,
    label_graph_rotation=None,
    tick_len=None,
    tick_gap=None,
    tick_text_shift=None,
    stand_values_side=None,
    reverse_scale=False,
    reverse_cmap=False,
):
    """Add a compact rotated RSSI scale anchored to a graph-side corner."""
    if fontsize is None:
        fontsize = IEEE_FONTSIZE
    use_auto_model_label_rotation = label_graph_rotation is None
    if scale_graph_side not in ("left", "right", "stand"):
        raise ValueError(f"Unsupported scale graph side: {scale_graph_side}")
    if stand_values_side is not None and stand_values_side not in ("left", "right"):
        raise ValueError(f"Unsupported stand values side: {stand_values_side}")

    if label_graph_side is None:
        if scale_graph_side == "left":
            label_graph_side = "right"
        elif scale_graph_side == "right":
            label_graph_side = "left"
        else:
            label_graph_side = "right"
    if label_graph_side not in ("left", "right"):
        raise ValueError(f"Unsupported label graph side: {label_graph_side}")
    loc_scale_xy = _resolve_xy_scale(loc_scale, "loc_scale")
    size_scale_xy = _resolve_xy_scale(size_scale, "size_scale")

    if scale_graph_side == "left":
        default_size = (0.525, 0.2)
        default_loc = (-0.075, 0.1)
        default_angle_deg = 240
        tick_sign = -1
        default_tick_len = 0.03
        default_tick_gap = 0.20
        default_tick_text_shift = (-0.15, -0.15)
        default_tick_text_ha = "left"
    elif scale_graph_side == "right":
        default_size = (0.525, 0.2)
        default_loc = (-0.0, 0.12)
        default_angle_deg = 151
        tick_sign = -1
        default_tick_len = 0.03
        default_tick_gap = 0.20
        default_tick_text_shift = (0.05, -0.2)
        default_tick_text_ha = "right"
    else: # Stand
        default_size = (0.05, 0.575)
        default_loc = (-0.075, -0.04)
        default_angle_deg = -5
        tick_sign = 0
        default_tick_len = 0.02
        default_tick_gap = 0.02
        default_tick_text_shift = (0.0, 0.0)
        default_tick_text_ha = "left"

    if scale_graph_side == "stand" and label_graph_side == "left":
        default_label_loc = (0.05, 0.0)
        default_label_rotation = 45.0
    elif scale_graph_side == "stand" and label_graph_side == "right":
        default_label_loc = (0.18, 0.0)
        default_label_rotation = 0.0
    elif scale_graph_side == "left" and label_graph_side == "left":
        default_label_loc = (-0.05, -0.125)
        default_label_rotation = 35.0
    elif scale_graph_side == "left" and label_graph_side == "right":
        default_label_loc = (-0.20, 0.0025)
        default_label_rotation = -12.5
    elif scale_graph_side == "right" and label_graph_side == "left":
        default_label_loc = (-0.0, -0.15)
        default_label_rotation = 35.0
    else:
        default_label_loc = (-0.05, 0.0)
        default_label_rotation = -15.0

    if loc is None:
        loc = default_loc
    else:
        loc = (float(loc[0]), float(loc[1]))
    if label_loc is None:
        label_loc = default_label_loc
    else:
        label_loc = (float(label_loc[0]), float(label_loc[1]))
    final_angle_deg = default_angle_deg if angle_deg is None else float(angle_deg)
    if tick_len is None:
        tick_len = default_tick_len
    if tick_gap is None:
        tick_gap = default_tick_gap
    if tick_text_shift is None:
        tick_text_shift = default_tick_text_shift
    else:
        tick_text_shift = (float(tick_text_shift[0]), float(tick_text_shift[1]))

    graph_bbox = _project_3d_graph_bbox_axes(fig, ax)
    ax_pos = ax.get_position()
    graph_w = max(graph_bbox[2] - graph_bbox[0], 1e-6)
    graph_h = max(graph_bbox[3] - graph_bbox[1], 1e-6)
    x0, x1 = ax.get_xlim3d()
    y0, _ = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()
    axis_origin = _project_3d_point_axes(fig, ax, x0, y0, z0)
    axis_x_end = _project_3d_point_axes(fig, ax, x1, y0, z0)
    axis_z_end = _project_3d_point_axes(fig, ax, x0, y0, z1)
    axis_xz_end = _project_3d_point_axes(fig, ax, x1, y0, z1)
    if axis_origin is None or axis_x_end is None or axis_z_end is None or axis_xz_end is None:
        axis_origin = np.array([graph_bbox[0], graph_bbox[1]], dtype=float)
        axis_x_end = axis_origin + np.array([1.0, 0.0], dtype=float)
        axis_z_end = axis_origin + np.array([0.0, 1.0], dtype=float)
        axis_xz_end = np.array([graph_bbox[2], graph_bbox[3]], dtype=float)
    x_dir = _normalize_2d(axis_x_end - axis_origin, fallback=(1.0, 0.0))
    z_dir = _normalize_2d(axis_z_end - axis_origin, fallback=(0.0, 1.0))
    top_left = axis_z_end
    top_right = axis_xz_end
    top_edge_dir = _normalize_2d(top_right - top_left, fallback=x_dir)
    if size is None:
        size = default_size
    size = _resolve_scale_size(size, graph_w, graph_h, ax, fig)
    size = (size[0] * size_scale_xy[0], size[1] * size_scale_xy[1])
    loc = (loc[0] * loc_scale_xy[0], loc[1] * loc_scale_xy[1])
    size_axes = (size[0] * graph_w, size[1] * graph_h)
    loc_offset = (loc[0] * graph_w, loc[1] * graph_h)
    if scale_graph_side == "left":
        base_loc = (graph_bbox[0], graph_bbox[3] - size_axes[1])
    elif scale_graph_side == "right":
        base_loc = (graph_bbox[2] - size_axes[0], graph_bbox[3] - size_axes[1])
    else:
        base_loc = (
            graph_bbox[0] - size_axes[0] - 0.015 * graph_w,
            graph_bbox[1] + 0.5 * (graph_h - size_axes[1]),
        )
    loc = (base_loc[0] + loc_offset[0], base_loc[1] + loc_offset[1])
    if scale_graph_side == "stand":
        scale_ax = fig.add_axes(ax_pos, zorder=20)
    else:
        scale_ax = fig.add_axes(
            [
                ax_pos.x0 + loc[0] * ax_pos.width,
                ax_pos.y0 + loc[1] * ax_pos.height,
                size_axes[0] * ax_pos.width,
                size_axes[1] * ax_pos.height,
            ],
            zorder=20,
        )
    scale_ax.set_xlim(0, 1)
    scale_ax.set_ylim(0, 1)
    scale_ax.axis("off")

    scale_color = DEBUG_3D_TEXT_COLOR if DEBUG_DARK_3D_BACKGROUND else "black"
    cmap_obj = cmap.reversed() if reverse_cmap else cmap
    if scale_graph_side == "stand":
        if stand_values_side is None:
            stand_values_side = "left" if label_graph_side == "right" else "right"
        z_axis_len = float(np.linalg.norm(axis_z_end - axis_origin))

        bar_width = size_axes[0]
        bar_height = size_axes[1]
        default_gap = 0.015 * graph_w
        center_offset = 0.5 * max(z_axis_len - bar_height, 0.0)
        bottom_right = (
            axis_origin
            + (loc_offset[0] - default_gap) * x_dir
            + (center_offset + loc_offset[1]) * z_dir
        )
        bottom_left = bottom_right - bar_width * x_dir
        affine = mtransforms.Affine2D.from_values(
            bar_width * x_dir[0],
            bar_width * x_dir[1],
            bar_height * z_dir[0],
            bar_height * z_dir[1],
            bottom_left[0],
            bottom_left[1],
        ) + scale_ax.transAxes

        if reverse_scale:
            gradient = np.linspace(vmax, vmin, 256).reshape(-1, 1)
        else:
            gradient = np.linspace(vmin, vmax, 256).reshape(-1, 1)
        scale_ax.imshow(
            gradient,
            extent=(0.0, 1.0, 0.0, 1.0),
            origin="lower",
            aspect="auto",
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
            transform=affine,
            clip_on=False,
            zorder=2,
        )
        scale_ax.add_patch(
            Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                fill=False,
                edgecolor=scale_color,
                linewidth=0.8,
                transform=affine,
                clip_on=False,
                zorder=3,
            )
        )

        for tick in _nice_rssi_ticks(vmin, vmax):
            if vmax == vmin:
                frac = 0.5
            elif reverse_scale:
                frac = 1.0 - (tick - vmin) / (vmax - vmin)
            else:
                frac = (tick - vmin) / (vmax - vmin)
            edge_point = bottom_left + frac * bar_height * z_dir
            if stand_values_side == "left":
                tick_start = edge_point
                tick_end = tick_start - tick_len * x_dir
                tick_text_pos = tick_end + (-tick_gap + tick_text_shift[0]) * x_dir + tick_text_shift[1] * z_dir
                tick_text_ha = "right"
            else:
                tick_start = edge_point + bar_width * x_dir
                tick_end = tick_start + tick_len * x_dir
                tick_text_pos = tick_end + (tick_gap + tick_text_shift[0]) * x_dir + tick_text_shift[1] * z_dir
                tick_text_ha = "left"
            scale_ax.plot(
                [tick_start[0], tick_end[0]],
                [tick_start[1], tick_end[1]],
                color=scale_color,
                linewidth=0.8,
                transform=scale_ax.transAxes,
                clip_on=False,
                zorder=4,
            )
            tick_text = f"{tick:.0f}" if abs(tick - round(tick)) < 1e-6 else f"{tick:.1f}"
            scale_ax.text(
                tick_text_pos[0],
                tick_text_pos[1],
                tick_text,
                transform=scale_ax.transAxes,
                ha=tick_text_ha,
                va="center",
                fontsize=fontsize,
                color=scale_color,
                clip_on=False,
                zorder=5,
            )

        existing_label = getattr(ax, "_rotated_scale_label_text", None)
        if existing_label is not None:
            try:
                existing_label.remove()
            except Exception:
                pass
            ax._rotated_scale_label_text = None

        top_left = bottom_left + bar_height * z_dir
        top_right = bottom_right + bar_height * z_dir
        top_edge_dir = _normalize_2d(top_right - top_left, fallback=x_dir)

        if use_auto_model_label_rotation:
            if label_graph_side == "left":
                label_graph_rotation = default_label_rotation
            else:
                label_graph_rotation = float(np.degrees(np.arctan2(top_edge_dir[1], top_edge_dir[0])))
        if label_graph_rotation is None:
            label_graph_rotation = default_label_rotation

        if label_graph_side == "left":
            label_anchor = top_left + label_loc[0] * bar_width * top_edge_dir + label_loc[1] * bar_height * z_dir
            label_ha = "left"
        else:
            label_anchor = top_right + label_loc[0] * bar_width * top_edge_dir + label_loc[1] * bar_height * z_dir
            label_ha = "left"
        ax._rotated_scale_label_text = ax.text2D(
            float(label_anchor[0]),
            float(label_anchor[1]),
            label,
            transform=ax.transAxes,
            rotation=label_graph_rotation,
            rotation_mode="anchor",
            ha=label_ha,
            va="bottom",
            fontsize=fontsize,
            color=scale_color,
        )
        return scale_ax

    grad_x0, grad_y0 = -0.0, 0.0
    grad_w, grad_h = 1.35, 0.10
    cx = grad_x0 + 0.4 * grad_w
    cy = grad_y0 + 0.8 * grad_h
    rot = mtransforms.Affine2D().rotate_deg_around(cx, cy, final_angle_deg) + scale_ax.transAxes

    gradient = np.linspace(vmin, vmax, 256).reshape(1, -1)
    scale_ax.imshow(
        gradient,
        extent=(grad_x0, grad_x0 + grad_w, grad_y0, grad_y0 + grad_h),
        origin="lower",
        aspect="auto",
        cmap=cmap_obj,
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
            edgecolor=scale_color,
            linewidth=0.8,
            transform=rot,
            clip_on=False,
            zorder=3,
        )
    )

    for tick in _nice_rssi_ticks(vmin, vmax):
        frac = 0.5 if vmax == vmin else (tick - vmin) / (vmax - vmin)
        x_tick = grad_x0 + frac * grad_w
        if tick_sign > 0:
            tick_y0 = grad_y0 + grad_h
            tick_y1 = grad_y0 + grad_h + tick_len
            tick_text_y = tick_y1 + tick_gap
            tick_text_va = "bottom"
        else:
            tick_y0 = grad_y0
            tick_y1 = grad_y0 - tick_len
            tick_text_y = tick_y1 - tick_gap
            tick_text_va = "top"
        tick_text_x = x_tick + tick_text_shift[0]
        tick_text_y += tick_text_shift[1]
        scale_ax.plot(
            [x_tick, x_tick],
            [tick_y0, tick_y1],
            color=scale_color,
            linewidth=0.8,
            transform=rot,
            clip_on=False,
            zorder=4,
        )
        tick_text = f"{tick:.0f}" if abs(tick - round(tick)) < 1e-6 else f"{tick:.1f}"
        scale_ax.text(
            tick_text_x,
            tick_text_y,
            tick_text,
            transform=rot,
            ha=default_tick_text_ha,
            va=tick_text_va,
            fontsize=fontsize,
            color=scale_color,
            clip_on=False,
            zorder=5,
        )

    existing_label = getattr(ax, "_rotated_scale_label_text", None)
    if existing_label is not None:
        try:
            existing_label.remove()
        except Exception:
            pass
        ax._rotated_scale_label_text = None

    if use_auto_model_label_rotation:
        label_graph_rotation = float(np.degrees(np.arctan2(top_edge_dir[1], top_edge_dir[0])))
    if label_graph_rotation is None:
        label_graph_rotation = default_label_rotation

    if label_graph_side == "left":
        label_anchor = top_left + label_loc[0] * graph_w * top_edge_dir + label_loc[1] * graph_h * z_dir
        label_ha = "left"
    else:
        label_anchor = top_right + label_loc[0] * graph_w * top_edge_dir + label_loc[1] * graph_h * z_dir
        label_ha = "right"
    ax._rotated_scale_label_text = ax.text2D(
        float(label_anchor[0]),
        float(label_anchor[1]),
        label,
        transform=ax.transAxes,
        rotation=label_graph_rotation,
        rotation_mode="anchor",
        ha=label_ha,
        va="top",
        fontsize=fontsize,
        color=scale_color,
    )
    return scale_ax


def _aggregate_3d_points(records, x_axis, y_axis, z_axis, metric_key="rssi_avg"):
    """Aggregate one metric over the unused dimension for a chosen 3D axis triplet."""
    grouped = defaultdict(list)
    for distance, sf, bw, tp, metrics in records:
        point_key = (
            _get_axis_val(x_axis, distance, sf, bw, tp),
            _get_axis_val(y_axis, distance, sf, bw, tp),
            _get_axis_val(z_axis, distance, sf, bw, tp),
        )
        metric_value = metrics.get(metric_key)
        if None in point_key or metric_value is None:
            continue
        grouped[point_key].append(float(metric_value))

    points = []
    for (x_val, y_val, z_val), values in grouped.items():
        points.append(
            {
                x_axis: x_val,
                y_axis: y_val,
                z_axis: z_val,
                metric_key: float(np.mean(values)),
            }
        )
    return points


def _clear_custom_z_label(ax):
    existing = getattr(ax, "_custom_zlabel_text", None)
    if existing is not None:
        text_items = existing if isinstance(existing, (list, tuple)) else [existing]
        for item in text_items:
            try:
                item.remove()
            except Exception:
                pass
        ax._custom_zlabel_text = None


def _format_custom_z_label(label, y_value):
    """Wrap longer overlay z-labels and nudge them upward slightly."""
    if "\n" in label:
        return label.splitlines(), y_value
    compact_label = label.replace(" ", "")
    if len(compact_label) <= 3:
        return [label], y_value
    if " (" in label:
        return label.replace(" (", "\n(").splitlines(), y_value + 0.05
    if " " in label:
        first, rest = label.split(" ", 1)
        return [first, rest], y_value + 0.05
    return [label], y_value


def _position_3d_z_label_top(ax, label, x=1.05, y=1.05, fontsize=IEEE_FONTSIZE, rotation_deg=0):
    """Draw a controllable z-label near the top of the z-axis using axes coordinates."""
    _clear_custom_z_label(ax)
    label_lines, y_pos = _format_custom_z_label(label, y)

    ax.set_zlabel("")
    line_step = 0.055
    text_items = []
    for idx, line in enumerate(label_lines):
        text_items.append(
            ax.text2D(
                x,
                y_pos - idx * line_step,
                line,
                transform=ax.transAxes,
                rotation=rotation_deg,
                rotation_mode="anchor",
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=DEBUG_3D_TEXT_COLOR if DEBUG_DARK_3D_BACKGROUND else "black",
            )
        )
    ax._custom_zlabel_text = text_items


def _resolve_axis_values(value):
    """Normalize a scalar or per-axis mapping/tuple to x/y/z values."""
    axes = ("x", "y", "z")
    if isinstance(value, dict):
        return {axis: float(value.get(axis, 0.0)) for axis in axes}
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return {axis: float(axis_value) for axis, axis_value in zip(axes, value)}
    scalar = float(value)
    return {axis: scalar for axis in axes}


def _plot_3d_scatter_panel(
    ax,
    points,
    x_axis,
    y_axis,
    z_axis,
    labels,
    rssi_min,
    rssi_max,
    bw_khz,
    dist_ticks,
    invert_x=False,
    invert_y=False,
    labelpad=0.0,
    tickpad=0.0,
    z_label_coords=None,
    z_label_rotation=0,
    view_elev=27,
    view_azim=-61,
    box_aspect=(1.0, 1.0, 0.90),
):
    """Render one reusable 3D scatter panel inside a combined figure."""
    if not points:
        ax.set_axis_off()
        return None
    label_pads = _resolve_axis_values(labelpad)
    tick_pads = _resolve_axis_values(tickpad)

    x_vals = np.array([point[x_axis] for point in points], dtype=float)
    y_vals = np.array([point[y_axis] for point in points], dtype=float)
    z_vals = np.array([point[z_axis] for point in points], dtype=float)
    rssi_vals = np.array([point["rssi_avg"] for point in points], dtype=float)

    if x_axis == "bw" or y_axis == "bw" or z_axis == "bw":
        for axis_name, axis_key in (("x", x_axis), ("y", y_axis), ("z", z_axis)):
            if axis_key != "bw":
                continue
            if axis_name == "x":
                x_vals = np.log10(np.maximum(x_vals, 1))
            elif axis_name == "y":
                y_vals = np.log10(np.maximum(y_vals, 1))
            else:
                z_vals = np.log10(np.maximum(z_vals, 1))
            break

    sc = ax.scatter(
        x_vals,
        y_vals,
        z_vals,
        c=rssi_vals,
        cmap="viridis",
        s=35,
        edgecolors="k",
        linewidths=0.3,
        vmin=rssi_min,
        vmax=rssi_max,
    )
    ax.view_init(elev=view_elev, azim=view_azim)
    try:
        ax.set_box_aspect(box_aspect)
    except Exception:
        pass
    ax.set_xlabel(labels[x_axis], fontsize=IEEE_FONTSIZE, labelpad=label_pads["x"])
    ax.set_ylabel(labels[y_axis], fontsize=IEEE_FONTSIZE, labelpad=label_pads["y"])
    if z_label_coords is None:
        _clear_custom_z_label(ax)
        ax.set_zlabel(labels[z_axis], fontsize=IEEE_FONTSIZE, labelpad=label_pads["z"])
        ax.zaxis.label.set_verticalalignment("top")
    else:
        _position_3d_z_label_top(
            ax,
            labels[z_axis],
            *z_label_coords,
            fontsize=IEEE_FONTSIZE,
            rotation_deg=z_label_rotation,
        )
    _set_3d_tick_pads(ax, tick_pads)

    for axis_name, axis_key in (("x", x_axis), ("y", y_axis), ("z", z_axis)):
        set_ticks = getattr(ax, f"set_{axis_name}ticks")
        set_ticklabels = getattr(ax, f"set_{axis_name}ticklabels")
        if axis_key == "sf":
            set_ticks(SF_VALUES)
            set_ticklabels([str(sf) for sf in SF_VALUES])
        elif axis_key == "bw":
            bw_ticks = np.log10(np.array(bw_khz))
            set_ticks(bw_ticks)
            set_ticklabels([_fmt_bw(bw) for bw in bw_khz])
        elif axis_key == "tp":
            set_ticks(TP_VALUES)
            set_ticklabels([str(tp) for tp in TP_VALUES])
        elif axis_key == "distance":
            set_ticks(dist_ticks)
            set_ticklabels([_fmt_distance(distance) for distance in dist_ticks])

    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    _apply_dark_3d_debug_theme(ax)
    return sc


def plot_rssi_3d(records, output_png, x_axis, y_axis, z_axis, invert_x=False, invert_y=False):
    """3D scatter: x_axis, y_axis, z_axis as axes, color=RSSI (avg). Aggregates over the 4th dimension."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    points = _aggregate_3d_points(records, x_axis, y_axis, z_axis, metric_key="rssi_avg")
    if not points:
        return
    rssi_vals = np.array([point["rssi_avg"] for point in points], dtype=float)
    fig = plt.figure(figsize=FIGSIZE_IEEE_DOUBLE)
    ax = fig.add_subplot(111, projection="3d")
    labels = {"sf": "SF", "bw": "BW (kHz)", "tp": "TP (dBm)", "distance": "Distance (m)"}
    bw_khz = sorted(set(bw / 1000.0 for bw in BW_VALUES))
    distances = sorted(set(d for d, _, _, _, _ in records))
    dist_ticks = _preferred_distance_ticks(distances)
    sc = _plot_3d_scatter_panel(
        ax,
        points,
        x_axis,
        y_axis,
        z_axis,
        labels,
        float(np.min(rssi_vals)),
        float(np.max(rssi_vals)),
        bw_khz,
        dist_ticks,
        invert_x=invert_x,
        invert_y=invert_y,
        labelpad=1.2,
        tickpad=0.6,
        z_label_coords=None,
    )
    fig.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05)
    _add_rotated_rssi_scale(
        fig,
        ax,
        sc.cmap,
        float(sc.norm.vmin),
        float(sc.norm.vmax),
        r"RSSI (avg) (dBm)",
        scale_graph_side="left",
        label_graph_side="right",
    )
    png_path, pdf_path = save_plot_outputs(fig, output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


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
    png_path, pdf_path = save_plot_outputs(fig, output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


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
    png_path, pdf_path = save_plot_outputs(fig, output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_rssi_3d_combined(records, output_png, scale_graph_side="left", label_graph_side="right"):
    """Three 3D subplots in one 2-column figure: A) BW-distance-SF, B) TP-distance-SF, C) TP-distance-BW."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    distances = sorted(set(d for d, _, _, _, _ in records))
    dist_ticks = _preferred_distance_ticks(distances)
    bw_khz = sorted(set(bw / 1000.0 for bw in BW_VALUES))
    labels = {"sf": "SF", "bw": "BW (kHz)", "tp": "TP (dBm)", "distance": "Distance (m)"}

    configs = [
        ("bw", "distance", "sf", True, False),
        ("tp", "distance", "sf", True, False),
        ("tp", "distance", "bw", True, False),
    ]
    panel_points = [
        (x_axis, y_axis, z_axis, inv_x, inv_y, _aggregate_3d_points(records, x_axis, y_axis, z_axis, metric_key="rssi_avg"))
        for x_axis, y_axis, z_axis, inv_x, inv_y in configs
    ]
    fig = plt.figure(figsize=FIGSIZE_IEEE_DOUBLE)
    rssi_all = []
    for _, _, _, _, _, points in panel_points:
        rssi_all.extend(point["rssi_avg"] for point in points)
    rssi_min, rssi_max = min(rssi_all), max(rssi_all) if rssi_all else (-100, -40)

    scatter_handles = []
    for idx, (x_axis, y_axis, z_axis, inv_x, inv_y, points) in enumerate(panel_points):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        z_label_coords = (1.00, 0.84) if idx == len(panel_points) - 1 else (1.02, 0.84)
        sc = _plot_3d_scatter_panel(
            ax,
            points,
            x_axis,
            y_axis,
            z_axis,
            labels,
            rssi_min,
            rssi_max,
            bw_khz,
            dist_ticks,
            invert_x=inv_x,
            invert_y=inv_y,
            labelpad={"x": -7.0, "y": -6.0, "z": 0.0},
            tickpad={"x": -4.0, "y": -3.0, "z": -1.0},
            z_label_coords=z_label_coords,
        )
        if sc is not None:
            scatter_handles.append((sc, ax, x_axis, y_axis, z_axis))

    fig.subplots_adjust(left=0.02, right=0.95, top=0.92, bottom=0.05, wspace=0.2)
    for sc, ax, _, _, _ in scatter_handles:
        _add_rotated_rssi_scale(
            fig,
            ax,
            sc.cmap,
            float(sc.norm.vmin),
            float(sc.norm.vmax),
            label=r"RSSI (dBm)",
            fontsize=IEEE_FONTSIZE,
            scale_graph_side=scale_graph_side,
            label_graph_side=label_graph_side,
        )
    png_path, pdf_path = save_plot_outputs(fig, output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


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
    plot_rssi_3d_combined(
        records,
        os.path.join(args.output_dir, "raw_rssi_3d_combined_right_left.png"),
        scale_graph_side="right",
        label_graph_side="left",
    )
    plot_rssi_3d_combined(
        records,
        os.path.join(args.output_dir, "raw_rssi_3d_combined_stand_left.png"),
        scale_graph_side="stand",
        label_graph_side="left",
    )
    plot_rssi_3d_combined(
        records,
        os.path.join(args.output_dir, "raw_rssi_3d_combined_stand_right.png"),
        scale_graph_side="stand",
        label_graph_side="right",
    )
    plot_rssi_distance_sfbw_surface(agg_records, os.path.join(args.output_dir, "raw_rssi_distance_sfbw_surface.png"))
    plot_rssi_distance_signal_tradeoff_surfaces(
        agg_records,
        os.path.join(args.output_dir, "raw_rssi_distance_signal_tradeoff.png"),
    )


if __name__ == "__main__":
    main()
