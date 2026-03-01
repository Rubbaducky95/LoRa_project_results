"""
Plot PER as heatmap in energy vs time for all configs combined.
Single heatmap aggregating all SF×BW×TP configurations.
Uses time_since_transmission_init_ms (continuous across distances) when available.
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
BG_DARK = "#fafafa"  # almost white axes background
from plot_per_vs_multiple_configs import (
    BW_VALUES,
    DATA_ROOT,
    SF_VALUES,
    TP_VALUES,
    WORKSPACE,
    config_matches_filter,
    file_packets,
    find_file_in_dist,
    get_file_order,
    parse_distance,
    setup_plot_style,
)


def _read_packets_with_tinit(path, payload_is_valid_fn):
    """Yield (time_since_transmission_init_ms, lost, energy) from CSV. Yields nothing if column missing."""
    try:
        rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    except Exception:
        return
    if not rows or "payload" not in rows[0]:
        return
    header = rows[0]
    if "time_since_transmission_init_ms" not in header:
        return
    tinit_idx = header.index("time_since_transmission_init_ms")
    payload_idx = header.index("payload")
    e_min_idx = header.index("energy_per_packet_min_mj") if "energy_per_packet_min_mj" in header else None
    e_max_idx = header.index("energy_per_packet_max_mj") if "energy_per_packet_max_mj" in header else None
    for r in rows[1:]:
        if len(r) <= payload_idx or str(r[payload_idx]).strip().startswith("CFG "):
            continue
        if len(r) <= tinit_idx or not r[tinit_idx]:
            continue
        try:
            t_ms = float(r[tinit_idx])
        except (ValueError, TypeError):
            continue
        lost = 0 if payload_is_valid_fn(r[payload_idx]) else 1
        energy = None
        if e_min_idx is not None and e_max_idx is not None and len(r) > max(e_min_idx, e_max_idx):
            try:
                energy = (float(r[e_min_idx]) + float(r[e_max_idx])) / 2.0
            except (ValueError, TypeError):
                pass
        yield (t_ms, lost, energy)


def collect_all_packets_time_energy_lost(data_root, filters=None, include_bw=None, include_sf=None, include_tp=None, collect_config_changes=False):
    """
    Returns [(time_min, energy_mj, lost), ...] for every packet.
    Uses time_since_transmission_init_ms when available. Time resets at each distance.
    include_bw/sf/tp: restrict to that value only.
    If collect_config_changes=True, also returns config_change_points: [(time_min, energy_mj, std_mj), ...]
    Time from first distance only; energy = mean, std = std dev across all distances.
    """
    from plot_per_vs_multiple_configs import payload_is_valid

    filters = filters or []
    pts = []
    config_times = {}  # (sf,bw,tp) -> time (first distance only)
    config_energies = {}  # (sf,bw,tp) -> list of first-packet energies (all distances)
    dist_folders = sorted(
        [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith("distance_")],
        key=lambda x: parse_distance(x) or 0,
    )
    first_dist = dist_folders[0] if (collect_config_changes and dist_folders) else None
    used_tinit = False

    for dn in dist_folders:
        dpath = os.path.join(data_root, dn)
        distance = parse_distance(dn)
        if distance is None:
            continue

        for sf, bw, tp in get_file_order():
            if include_bw is not None and bw != include_bw:
                continue
            if include_sf is not None and sf != include_sf:
                continue
            if include_tp is not None and tp != include_tp:
                continue
            if any(config_matches_filter(sf, bw, tp, distance, f) for f in filters):
                continue
            path = find_file_in_dist(dpath, sf, bw, tp)
            if not path or not os.path.isfile(path):
                continue

            first_packet_recorded = False
            for t_ms, lost, energy in _read_packets_with_tinit(path, payload_is_valid):
                used_tinit = True
                if energy is not None and energy > 0:
                    time_min = t_ms / 60000.0  # resets per distance (T_init is per-distance)
                    pts.append((time_min, energy, lost))
                    if collect_config_changes and not first_packet_recorded:
                        key = (sf, bw, tp)
                        if dn == first_dist:
                            config_times[key] = time_min
                        if key not in config_energies:
                            config_energies[key] = []
                        config_energies[key].append(energy)
                        first_packet_recorded = True

    if not used_tinit:
        # Fallback: original timer logic (no T_init column)
        pts = []
        config_times = {}
        config_energies = {}
        timer_ms = 0
        prev_raw_ms = None
        prev_distance = None
        for dn in dist_folders:
            dpath = os.path.join(data_root, dn)
            distance = parse_distance(dn)
            if distance is None:
                continue
            if prev_distance is not None:
                timer_ms = 0
                prev_raw_ms = None
            prev_distance = distance
            for sf, bw, tp in get_file_order():
                if include_bw is not None and bw != include_bw:
                    continue
                if include_sf is not None and sf != include_sf:
                    continue
                if include_tp is not None and tp != include_tp:
                    continue
                if any(config_matches_filter(sf, bw, tp, distance, f) for f in filters):
                    continue
                path = find_file_in_dist(dpath, sf, bw, tp)
                if not path or not os.path.isfile(path):
                    continue
                first_packet_recorded = False
                for raw_ms, lost, _, energy, tx_ms in file_packets(path, distance, sf, bw, tp):
                    if prev_raw_ms is not None and raw_ms < prev_raw_ms:
                        timer_ms = 0
                    time_min = timer_ms / 60000.0
                    if energy is not None and energy > 0:
                        pts.append((time_min, energy, lost))
                        if collect_config_changes and not first_packet_recorded:
                            key = (sf, bw, tp)
                            if dn == first_dist:
                                config_times[key] = time_min
                            if key not in config_energies:
                                config_energies[key] = []
                            config_energies[key].append(energy)
                            first_packet_recorded = True
                    timer_ms += tx_ms
                    prev_raw_ms = raw_ms

    if collect_config_changes:
        config_change_points = []
        for sf, bw, tp in get_file_order():
            key = (sf, bw, tp)
            if key in config_times and key in config_energies and config_energies[key]:
                t = config_times[key]
                energies = config_energies[key]
                e_mean = float(np.mean(energies))
                e_std = float(np.std(energies)) if len(energies) > 1 else 0.0
                config_change_points.append((t, e_mean, e_std))
        return pts, config_change_points
    return pts


def build_heatmap_matrix_per_packet(pts, time_bins, energy_bins):
    """Bin (time, energy, lost) per packet. PER per cell = 100 * sum(lost) / count."""
    mat = np.full((len(energy_bins) - 1, len(time_bins) - 1), np.nan)
    lost_sums = np.zeros_like(mat)
    counts = np.zeros_like(mat)
    for t, e, lost in pts:
        ti = np.searchsorted(time_bins, t, side="right") - 1
        ei = np.searchsorted(energy_bins, e, side="right") - 1
        if 0 <= ti < mat.shape[1] and 0 <= ei < mat.shape[0]:
            lost_sums[ei, ti] += lost
            counts[ei, ti] += 1
    valid = counts > 0
    mat[valid] = 100.0 * lost_sums[valid] / counts[valid]
    return mat


def main():
    setup_plot_style()
    output_dir = os.path.join(WORKSPACE, "results", "raw_test_data_plots")
    os.makedirs(output_dir, exist_ok=True)
    filters = []
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color=BG_DARK)
    n_time_bins = 30
    n_energy_bins = 30

    result = collect_all_packets_time_energy_lost(DATA_ROOT, filters, collect_config_changes=True)
    pts, config_change_points = result
    if not pts:
        print("No data found.")
        return

    all_times = [p[0] for p in pts]
    all_energies = [p[1] for p in pts]
    per_vals = [100.0 * p[2] for p in pts]  # per-packet: 0 or 100
    t_min, t_max = min(all_times), max(all_times)
    e_min, e_max = min(all_energies), max(all_energies)
    t_min = max(0, t_min - 0.5)
    t_max = t_max + 0.5
    e_min = max(0.01, e_min * 0.99)
    e_max = e_max * 1.01
    time_bins = np.linspace(t_min, t_max, n_time_bins + 1)
    energy_bins = np.linspace(e_min, e_max, n_energy_bins + 1)

    vmin = 0
    vmax = 20  # scale 0-100 PER to 0-20 for display (log scale)
    log_vmin = 0.01  # avoid log(0)

    mat = build_heatmap_matrix_per_packet(pts, time_bins, energy_bins)

    # Pure energy vs time PER heatmap (disabled for now)
    # fig, ax = plt.subplots(figsize=(8, 8))
    # ax.set_facecolor(BG_DARK)
    # ax.tick_params(axis="both", colors="black")
    # ax.xaxis.label.set_color("black")
    # ax.yaxis.label.set_color("black")
    # ax.title.set_color("black")
    # for spine in ax.spines.values():
    #     spine.set_color("black")
    # im = ax.pcolormesh(
    #     time_bins,
    #     energy_bins,
    #     mat,
    #     cmap=cmap,
    #     vmin=vmin,
    #     vmax=vmax,
    #     shading="flat",
    # )
    # ax.set_xlim(t_min, t_max)
    # ax.set_ylim(e_min, e_max)
    # ax.set_aspect((t_max - t_min) / (e_max - e_min))  # square cells
    # ax.set_xlabel("Time (minutes, time since first packet sent)")
    # ax.set_ylabel("Energy per packet (mJ)")
    # ax.set_title(f"PER heatmap: energy vs time ({len(pts):,} packets)")
    # fig.tight_layout()
    # cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    # cbar.set_label("PER (%)")
    # out_path = os.path.join(output_dir, "raw_per_gradient_energy_vs_time.png")
    # fig.savefig(out_path, dpi=220, bbox_inches="tight")
    # plt.close()
    # print(f"Saved: raw_per_gradient_energy_vs_time.png ({len(pts):,} packets)")

    # Param transitions: one color-to-black cmap per value (BW 62.5, 125, etc.)
    plot_param_transitions_bw(
        output_dir, DATA_ROOT, filters, time_bins, energy_bins,
        t_min, t_max, e_min, e_max, vmin, vmax, log_vmin, build_heatmap_matrix_per_packet,
        config_change_points,
    )
    plot_param_transitions_sf(
        output_dir, DATA_ROOT, filters, time_bins, energy_bins,
        t_min, t_max, e_min, e_max, vmin, vmax, log_vmin, build_heatmap_matrix_per_packet,
        config_change_points,
    )
    plot_param_transitions_tp(
        output_dir, DATA_ROOT, filters, time_bins, energy_bins,
        t_min, t_max, e_min, e_max, vmin, vmax, log_vmin, build_heatmap_matrix_per_packet,
        config_change_points,
    )


def _plot_config_change_points(ax, config_change_points, t_min, t_max, e_min, e_max):
    """Plot opaque circles at config change (time, energy), radius = std dev of energy."""
    if not config_change_points:
        return
    for item in config_change_points:
        if len(item) == 3:
            t, e, std = item
        else:
            t, e = item
            std = 0.0
        if not (t_min <= t <= t_max and e_min <= e <= e_max):
            continue
        circ = mpatches.Circle(
            (t, e), radius=max(std, 0.001), transform=ax.transData,
            fill=True, facecolor="black", edgecolor="black", alpha=1, zorder=10,
        )
        ax.add_patch(circ)


def _truncate_cmap_no_white(cmap, minval=0.25):
    """Truncate colormap so 0 (vmin) maps to a visible color, not white."""
    return mcolors.LinearSegmentedColormap.from_list("truncated", cmap(np.linspace(minval, 1, 256)))


def _color_to_black_cmap(base_cmap, name):
    """Create cmap from base color to dark tint (not pure black) so configs stay distinguishable."""
    color = base_cmap(0.65)  # characteristic color from the cmap
    dark = base_cmap(0.95)   # dark tint of same color (not pure black)
    cm = mcolors.LinearSegmentedColormap.from_list(name, [color, dark])
    cm.set_bad(color="none", alpha=0)
    return cm


# Base colormaps for per-value color-to-black (Blues, Reds, Greens, Purples, Oranges, RdPu for SF)
_BASE_CMAPS = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges, plt.cm.RdPu]


def _legend_color(base_cmap):
    """Characteristic color from cmap (matches color-to-black start)."""
    return base_cmap(0.65)


def _add_legend_labels(ax, labels):
    """Draw labels only, centered under each colorbar, rotated 90 degrees."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(labels)
    if n == 0:
        return
    for i in range(n):
        x = (i + 0.5) / n
        ax.text(x, 0.25, labels[i], ha="center", va="center", fontsize=7, transform=ax.transAxes, rotation=90)


def _tighten_scale_label_gap(fig, n_cbars, gap_reduce=0.05):
    """Manually move colorbar axes down and legend axes up to reduce gap."""
    cbar_axes = fig.axes[1:1 + n_cbars]
    ax_legend = fig.axes[1 + n_cbars]
    leg_pos = ax_legend.get_position()
    for cax in cbar_axes:
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 - gap_reduce, pos.width, pos.height + gap_reduce])
    ax_legend.set_position([leg_pos.x0, leg_pos.y0 + gap_reduce, leg_pos.width, leg_pos.height])


def _shift_scale_labels_down(fig, n_cbars, shift=0.05):
    """Move the whole scale+labels block down."""
    cbar_axes = fig.axes[1:1 + n_cbars]
    ax_legend = fig.axes[1 + n_cbars]
    for cax in cbar_axes:
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 - shift, pos.width, pos.height])
    leg_pos = ax_legend.get_position()
    ax_legend.set_position([leg_pos.x0, leg_pos.y0 - shift, leg_pos.width, leg_pos.height])


def _add_param_colorbars(fig, gs_cbars, values, log_vmin, vmax):
    """Add one thin vertical colorbar per value, side by side on the right. Log scale."""
    n = len(values)
    norm = LogNorm(vmin=log_vmin, vmax=vmax)
    for i in range(n):
        cax = fig.add_subplot(gs_cbars[i])
        cmap = _color_to_black_cmap(_BASE_CMAPS[i % len(_BASE_CMAPS)], f"cb_{i}")
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
        if i < n - 1:
            cbar.ax.set_yticks([])
            cbar.ax.set_yticklabels([])
        else:
            cbar.ax.tick_params(labelsize=7)


def plot_param_transitions_bw(output_dir, data_root, filters, time_bins, energy_bins, t_min, t_max, e_min, e_max, vmin, vmax, log_vmin, build_matrix_fn, config_change_points=None):
    """Overlay all BW heatmaps, one color-to-black cmap per BW value."""
    fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.18], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(BG_DARK)
    ax.tick_params(axis="both", colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")
    for idx, bw in enumerate(BW_VALUES):
        cmap = _color_to_black_cmap(_BASE_CMAPS[idx % len(_BASE_CMAPS)], f"bw_{bw}")
        pts_b = collect_all_packets_time_energy_lost(data_root, filters, include_bw=bw)
        mat = build_matrix_fn(pts_b, time_bins, energy_bins)
        scaled = np.where(np.isnan(mat), np.nan, np.maximum(mat * 0.2, log_vmin))
        mat_masked = np.ma.masked_where(np.isnan(mat), scaled)
        im = ax.pcolormesh(
            time_bins, energy_bins, mat_masked,
            cmap=cmap, norm=LogNorm(vmin=log_vmin, vmax=vmax), shading="flat",
        )
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(e_min, e_max)
    ax.set_aspect((t_max - t_min) / (e_max - e_min))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Energy (mJ)")
    _plot_config_change_points(ax, config_change_points or [], t_min, t_max, e_min, e_max)
    labels_bw = [f"BW{bw/1000:.1f}kHz".replace(".0", "") for bw in BW_VALUES]
    gs_right = gs[1].subgridspec(2, 1, height_ratios=[0.72, 0.28], hspace=0)
    gs_cbars = gs_right[0].subgridspec(1, len(BW_VALUES), wspace=0.08)
    _add_param_colorbars(fig, gs_cbars, BW_VALUES, log_vmin, vmax)
    ax_leg = fig.add_subplot(gs_right[1])
    _add_legend_labels(ax_leg, labels_bw)
    _tighten_scale_label_gap(fig, len(BW_VALUES))
    _shift_scale_labels_down(fig, len(BW_VALUES))
    fig.savefig(os.path.join(output_dir, "raw_per_gradient_energy_vs_time_bw_transitions.png"), dpi=220, bbox_inches="tight")
    plt.close()
    print("Saved: raw_per_gradient_energy_vs_time_bw_transitions.png")


def plot_param_transitions_sf(output_dir, data_root, filters, time_bins, energy_bins, t_min, t_max, e_min, e_max, vmin, vmax, log_vmin, build_matrix_fn, config_change_points=None):
    """Overlay all SF heatmaps, one color-to-black cmap per SF value."""
    fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.18], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(BG_DARK)
    ax.tick_params(axis="both", colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")
    for idx, sf in enumerate(SF_VALUES):
        cmap = _color_to_black_cmap(_BASE_CMAPS[idx % len(_BASE_CMAPS)], f"sf_{sf}")
        pts_s = collect_all_packets_time_energy_lost(data_root, filters, include_sf=sf)
        mat = build_matrix_fn(pts_s, time_bins, energy_bins)
        scaled = np.where(np.isnan(mat), np.nan, np.maximum(mat * 0.2, log_vmin))
        mat_masked = np.ma.masked_where(np.isnan(mat), scaled)
        im = ax.pcolormesh(
            time_bins, energy_bins, mat_masked,
            cmap=cmap, norm=LogNorm(vmin=log_vmin, vmax=vmax), shading="flat",
        )
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(e_min, e_max)
    ax.set_aspect((t_max - t_min) / (e_max - e_min))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Energy (mJ)")
    _plot_config_change_points(ax, config_change_points or [], t_min, t_max, e_min, e_max)
    labels_sf = [f"SF{sf}" for sf in SF_VALUES]  # no space: SF7 not SF 7
    gs_right = gs[1].subgridspec(2, 1, height_ratios=[0.72, 0.28], hspace=0)
    gs_cbars = gs_right[0].subgridspec(1, len(SF_VALUES), wspace=0.05)
    _add_param_colorbars(fig, gs_cbars, SF_VALUES, log_vmin, vmax)
    ax_leg = fig.add_subplot(gs_right[1])
    _add_legend_labels(ax_leg, labels_sf)
    _tighten_scale_label_gap(fig, len(SF_VALUES))
    _shift_scale_labels_down(fig, len(SF_VALUES))
    fig.savefig(os.path.join(output_dir, "raw_per_gradient_energy_vs_time_sf_transitions.png"), dpi=220, bbox_inches="tight")
    plt.close()
    print("Saved: raw_per_gradient_energy_vs_time_sf_transitions.png")


def plot_param_transitions_tp(output_dir, data_root, filters, time_bins, energy_bins, t_min, t_max, e_min, e_max, vmin, vmax, log_vmin, build_matrix_fn, config_change_points=None):
    """Overlay all TP heatmaps, one color-to-black cmap per TP value."""
    fig = plt.figure(figsize=(9, 8))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.18], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(BG_DARK)
    ax.tick_params(axis="both", colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")
    for idx, tp in enumerate(TP_VALUES):
        cmap = _color_to_black_cmap(_BASE_CMAPS[idx % len(_BASE_CMAPS)], f"tp_{tp}")
        pts_t = collect_all_packets_time_energy_lost(data_root, filters, include_tp=tp)
        mat = build_matrix_fn(pts_t, time_bins, energy_bins)
        scaled = np.where(np.isnan(mat), np.nan, np.maximum(mat * 0.2, log_vmin))
        mat_masked = np.ma.masked_where(np.isnan(mat), scaled)
        im = ax.pcolormesh(
            time_bins, energy_bins, mat_masked,
            cmap=cmap, norm=LogNorm(vmin=log_vmin, vmax=vmax), shading="flat",
        )
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(e_min, e_max)
    ax.set_aspect((t_max - t_min) / (e_max - e_min))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Energy (mJ)")
    _plot_config_change_points(ax, config_change_points or [], t_min, t_max, e_min, e_max)
    labels_tp = [f"TP{tp}" for tp in TP_VALUES]
    gs_right = gs[1].subgridspec(2, 1, height_ratios=[0.72, 0.28], hspace=0)
    gs_cbars = gs_right[0].subgridspec(1, len(TP_VALUES), wspace=0.1)
    _add_param_colorbars(fig, gs_cbars, TP_VALUES, log_vmin, vmax)
    ax_leg = fig.add_subplot(gs_right[1])
    _add_legend_labels(ax_leg, labels_tp)
    _tighten_scale_label_gap(fig, len(TP_VALUES))
    _shift_scale_labels_down(fig, len(TP_VALUES))
    fig.savefig(os.path.join(output_dir, "raw_per_gradient_energy_vs_time_tp_transitions.png"), dpi=220, bbox_inches="tight")
    plt.close()
    print("Saved: raw_per_gradient_energy_vs_time_tp_transitions.png")


if __name__ == "__main__":
    main()
