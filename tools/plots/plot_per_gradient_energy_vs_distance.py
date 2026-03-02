"""
Plot PER as heatmap in energy vs distance for all configs combined.
Separate script so the time-based script only generates time-based plots.
"""
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from plot_per_gradient_energy_vs_time import (
    BG_DARK,
    IEEE_FONTSIZE,
    _add_legend_labels,
    _add_param_colorbars,
    _add_per_label_above_colorbars,
    _align_colorbars_to_plot,
    _color_to_black_cmap,
    _enforce_axis_fontsize,
    _legend_values_and_labels,
    _normalize_figure_fonts,
    _pull_scales_closer,
    _shift_scale_labels_down,
    _tighten_scale_label_gap,
    _BASE_CMAPS,
    build_heatmap_matrix_per_packet,
)
from plot_per_vs_multiple_configs import (
    DATA_ROOT,
    WORKSPACE,
    config_matches_filter,
    file_packets,
    find_file_in_dist,
    get_file_order,
    parse_distance,
    setup_plot_style,
)


def collect_all_packets_distance_energy_lost(data_root, filters=None, include_bw=None, include_sf=None, include_tp=None):
    from plot_per_vs_multiple_configs import payload_is_valid

    filters = filters or []
    pts = []
    dist_folders = sorted(
        [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith("distance_")],
        key=lambda x: parse_distance(x) or 0,
    )

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

            used_tinit = False
            for _, lost, energy in _read_packets_with_tinit(path, payload_is_valid):
                used_tinit = True
                if energy is not None and energy > 0:
                    pts.append((distance, energy, lost))

            if used_tinit:
                continue

            for _, lost, _, energy, _ in file_packets(path, distance, sf, bw, tp):
                if energy is not None and energy > 0:
                    pts.append((distance, energy, lost))

    return pts


def _read_packets_with_tinit(path, payload_is_valid_fn):
    try:
        import csv

        rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    except Exception:
        return
    if not rows or "payload" not in rows[0]:
        return
    header = rows[0]
    if "time_since_transmission_init_ms" not in header:
        return
    payload_idx = header.index("payload")
    e_min_idx = header.index("energy_per_packet_min_mj") if "energy_per_packet_min_mj" in header else None
    e_max_idx = header.index("energy_per_packet_max_mj") if "energy_per_packet_max_mj" in header else None
    for r in rows[1:]:
        if len(r) <= payload_idx or str(r[payload_idx]).strip().startswith("CFG "):
            continue
        lost = 0 if payload_is_valid_fn(r[payload_idx]) else 1
        energy = None
        if e_min_idx is not None and e_max_idx is not None and len(r) > max(e_min_idx, e_max_idx):
            try:
                energy = (float(r[e_min_idx]) + float(r[e_max_idx])) / 2.0
            except (ValueError, TypeError):
                pass
        yield (0.0, lost, energy)


def _bins_from_centers(values):
    values = sorted(set(values))
    if not values:
        return np.array([])
    if len(values) == 1:
        half_step = 0.5
        return np.array([values[0] - half_step, values[0] + half_step], dtype=float)
    mids = [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]
    first_edge = values[0] - (mids[0] - values[0])
    last_edge = values[-1] + (values[-1] - mids[-1])
    return np.array([first_edge, *mids, last_edge], dtype=float)


def _set_distance_axis(ax, distances, d_min, d_max):
    ax.set_xlim(d_min, d_max)
    ax.set_xlabel("Distance (m)", fontsize=IEEE_FONTSIZE)
    ax.set_xticks(distances)
    ax.set_xticklabels([f"{d:.2f}".rstrip("0").rstrip(".") for d in distances], rotation=45, ha="right")


def plot_distance_heatmap(output_dir, distance_bins, distances, energy_bins, mat, d_min, d_max, e_min, e_max, vmax, log_vmin, cmap, suffix=""):
    fig, ax = plt.subplots(figsize=(7.16, 4))
    ax.set_facecolor(BG_DARK)
    ax.tick_params(axis="both", colors="black", labelsize=IEEE_FONTSIZE)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")

    scaled = np.where(np.isnan(mat), np.nan, np.maximum(mat * 0.2, log_vmin))
    mat_masked = np.ma.masked_where(np.isnan(mat), scaled)
    im = ax.pcolormesh(
        distance_bins, energy_bins, mat_masked,
        cmap=cmap, norm=LogNorm(vmin=log_vmin, vmax=vmax), shading="flat",
    )
    _set_distance_axis(ax, distances, d_min, d_max)
    ax.set_ylim(e_min, e_max)
    ax.set_ylabel("Energy (mJ)", fontsize=IEEE_FONTSIZE)
    _enforce_axis_fontsize(ax)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=IEEE_FONTSIZE)
    cbar.set_label("PER (%)", fontsize=IEEE_FONTSIZE)
    _normalize_figure_fonts(fig)
    fname = f"raw_per_gradient_energy_vs_distance{suffix}.png" if suffix else "raw_per_gradient_energy_vs_distance.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_distance_param_overlay(output_dir, data_root, filters, distance_bins, distances, energy_bins, d_min, d_max, e_min, e_max, vmax, log_vmin, param_type, output_suffix=""):
    values, labels = _legend_values_and_labels(param_type)
    if not values:
        return

    fig = plt.figure(figsize=(7.16, 4))
    gs = plt.matplotlib.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.12], wspace=0.001)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(BG_DARK)
    ax.tick_params(axis="both", colors="black", labelsize=IEEE_FONTSIZE)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    for spine in ax.spines.values():
        spine.set_color("black")

    collect_kwargs = {"bw": "include_bw", "sf": "include_sf", "tp": "include_tp"}
    key_name = collect_kwargs[param_type]
    for idx, value in enumerate(values):
        cmap_value = _color_to_black_cmap(_BASE_CMAPS[idx % len(_BASE_CMAPS)], f"{param_type}_{value}")
        kwargs = {key_name: value}
        pts_value = collect_all_packets_distance_energy_lost(data_root, filters, **kwargs)
        if not pts_value:
            continue
        mat = build_heatmap_matrix_per_packet(pts_value, distance_bins, energy_bins)
        scaled = np.where(np.isnan(mat), np.nan, np.maximum(mat * 0.2, log_vmin))
        mat_masked = np.ma.masked_where(np.isnan(mat), scaled)
        ax.pcolormesh(
            distance_bins, energy_bins, mat_masked,
            cmap=cmap_value, norm=LogNorm(vmin=log_vmin, vmax=vmax), shading="flat",
        )

    _set_distance_axis(ax, distances, d_min, d_max)
    ax.set_ylim(e_min, e_max)
    ax.set_ylabel("Energy (mJ)", fontsize=IEEE_FONTSIZE)
    _enforce_axis_fontsize(ax)

    prefix = {"bw": "BW", "sf": "SF", "tp": "TP"}[param_type]
    labels_full = [f"{prefix}{label}" for label in labels]
    unit = "kHz" if param_type == "bw" else ("dBm" if param_type == "tp" else None)

    gs_right = gs[1].subgridspec(2, 1, height_ratios=[0.72, 0.28], hspace=0)
    gs_cbars = gs_right[0].subgridspec(1, len(values), wspace=0.02 if len(values) <= 4 else 0.03)
    _add_param_colorbars(fig, gs_cbars, values, log_vmin, vmax)
    ax_leg = fig.add_subplot(gs_right[1])
    _add_legend_labels(ax_leg, labels_full, unit=unit, pad=0.03 if param_type == "sf" else 0, unit_y=-0.25 if unit else -0.15)
    _tighten_scale_label_gap(fig, len(values))
    _shift_scale_labels_down(fig, len(values))
    _pull_scales_closer(fig, len(values))
    _align_colorbars_to_plot(fig, len(values))
    _add_per_label_above_colorbars(fig, len(values))
    _normalize_figure_fonts(fig)
    fname = f"raw_per_gradient_energy_vs_distance_{param_type}_layers{output_suffix}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def main():
    setup_plot_style()
    plt.rcParams.update(
        {
            "font.size": IEEE_FONTSIZE,
            "axes.labelsize": IEEE_FONTSIZE,
            "axes.titlesize": IEEE_FONTSIZE,
            "xtick.labelsize": IEEE_FONTSIZE,
            "ytick.labelsize": IEEE_FONTSIZE,
            "legend.fontsize": IEEE_FONTSIZE,
            "legend.title_fontsize": IEEE_FONTSIZE,
            "figure.titlesize": IEEE_FONTSIZE,
            "axes.unicode_minus": False,
        }
    )

    version = "v2"
    output_dir = os.path.join(WORKSPACE, "results", "raw_test_data_plots", "energy_vs_distance", version)
    os.makedirs(output_dir, exist_ok=True)
    filters = []
    cmap = mcolors.LinearSegmentedColormap.from_list("per_viridis", plt.cm.viridis_r(np.linspace(0.1, 1, 256)), N=256)
    cmap.set_bad(color=BG_DARK)
    n_energy_bins = 30
    vmax = 20
    log_vmin = 0.01

    pts = collect_all_packets_distance_energy_lost(DATA_ROOT, filters)
    if not pts:
        print("No data found.")
        return

    distances = sorted(set(p[0] for p in pts))
    all_energies = [p[1] for p in pts]
    distance_bins = _bins_from_centers(distances)
    d_min, d_max = distance_bins[0], distance_bins[-1]
    e_min = max(0.01, min(all_energies) * 0.99)
    e_max = max(all_energies) * 1.01
    energy_bins = np.linspace(e_min, e_max, n_energy_bins + 1)
    mat = build_heatmap_matrix_per_packet(pts, distance_bins, energy_bins)

    plot_distance_heatmap(output_dir, distance_bins, distances, energy_bins, mat, d_min, d_max, e_min, e_max, vmax, log_vmin, cmap, suffix=f"_{version}")
    plot_distance_param_overlay(output_dir, DATA_ROOT, filters, distance_bins, distances, energy_bins, d_min, d_max, e_min, e_max, vmax, log_vmin, "bw", output_suffix=f"_{version}")
    plot_distance_param_overlay(output_dir, DATA_ROOT, filters, distance_bins, distances, energy_bins, d_min, d_max, e_min, e_max, vmax, log_vmin, "sf", output_suffix=f"_{version}")
    plot_distance_param_overlay(output_dir, DATA_ROOT, filters, distance_bins, distances, energy_bins, d_min, d_max, e_min, e_max, vmax, log_vmin, "tp", output_suffix=f"_{version}")


if __name__ == "__main__":
    main()
