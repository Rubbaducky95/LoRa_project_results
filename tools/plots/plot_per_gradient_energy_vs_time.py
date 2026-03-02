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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib import patheffects
from matplotlib.text import Text
from matplotlib.transforms import blended_transform_factory

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
BG_DARK = "#e8e8e8"  # light grey axes background

from plot_config import IEEE_FONTSIZE, FIGSIZE_TWO_COL, FIGSIZE_ONE_COL, SAVE_DPI
# Smaller font for config switch markers (S0, B0, etc.) inside the plot
MARKER_FONTSIZE = 7
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


def _parse_config_csv_row(row):
    """Parse config from CSV row. Returns (time_min, sf, bw, tp) or None. Config format: '7, 62.5' -> SF=7, BW=62.5kHz."""
    try:
        t_s = float(row.get("T_init_s", 0))
        time_min = t_s / 60.0
        cfg = row.get("config", "").strip().strip('"')
        tp = int(row.get("TP", 0))
        parts = [p.strip() for p in cfg.split(",")]
        if len(parts) != 2:
            return None
        sf = int(parts[0])
        bw_khz = float(parts[1])
        bw = int(bw_khz * 1000)
        return (time_min, sf, bw, tp)
    except (ValueError, TypeError):
        return None


def _load_config_switches_from_csv(output_dir):
    """Load config switch times and configs from config_change_T_init.csv. Returns [(time_min, sf, bw, tp), ...]."""
    path = os.path.join(output_dir, "config_change_T_init.csv")
    if not os.path.isfile(path):
        return []
    switches = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = _parse_config_csv_row(row)
            if parsed is not None:
                switches.append(parsed)
    return switches


def _config_avg_energy(data_root, filters, sf, bw, tp):
    """Return average energy (mJ) for config (sf, bw, tp) across all distances."""
    pts = collect_all_packets_time_energy_lost(
        data_root, filters or [], include_sf=sf, include_bw=bw, include_tp=tp
    )
    energies = [p[1] for p in pts if p[1] is not None and p[1] > 0]
    return float(np.mean(energies)) if energies else None


def _get_all_config_switch_points(output_dir, data_root, filters, include_params=None,
                                   time_min_threshold=0, energy_min_threshold=0):
    """
    Return [(time_min, avg_energy_mj, param_type), ...] for config switches.
    param_type: 'sf'|'bw'|'tp' which param changed, or None if include_params is None.
    include_params: list like ['sf','tp'] - only include switches where any of those params changed.
    time_min_threshold, energy_min_threshold: include if time > t OR energy > e.
    """
    switches = _load_config_switches_from_csv(output_dir)
    if not switches:
        return []
    if include_params is not None and isinstance(include_params, str):
        include_params = [include_params]
    filters = filters or []
    points = []
    prev_sf = prev_bw = prev_tp = None
    for time_min, sf, bw, tp in switches:
        param_type = None
        if include_params is not None and prev_sf is not None:
            sf_changed = sf != prev_sf
            bw_changed = bw != prev_bw
            tp_changed = tp != prev_tp
            if sf_changed and "sf" in include_params:
                param_type = "sf"
            elif bw_changed and "bw" in include_params:
                param_type = "bw"
            elif tp_changed and "tp" in include_params:
                param_type = "tp"
            else:
                prev_sf, prev_bw, prev_tp = sf, bw, tp
                continue
        prev_sf, prev_bw, prev_tp = sf, bw, tp
        avg_e = _config_avg_energy(data_root, filters, sf, bw, tp)
        if avg_e is not None and (time_min > time_min_threshold or avg_e > energy_min_threshold):
            param_val = sf if param_type == "sf" else (bw if param_type == "bw" else (tp if param_type == "tp" else None))
            points.append((float(time_min), avg_e, param_type, param_val, sf, bw, tp))
    return points


_CONFIG_SWITCH_RANGE = {"sf": "SF", "bw": "BW", "tp": "TP"}

# --- Size-based markers (commented out, was best option) ---
# def _marker_size_from_param(param_type, param_val):
#     """Size increases with parameter. Normalized so max SF = max BW = max TP."""
#     size_min, size_max = 5, 9
#     if param_val is None:
#         return (size_min + size_max) / 2
#     if param_type == "sf":
#         idx = SF_VALUES.index(param_val) if param_val in SF_VALUES else 0
#         n = len(SF_VALUES)
#         frac = idx / (n - 1) if n > 1 else 0
#     elif param_type == "bw":
#         idx = BW_VALUES.index(param_val) if param_val in BW_VALUES else 0
#         n = len(BW_VALUES)
#         frac = idx / (n - 1) if n > 1 else 0
#     elif param_type == "tp":
#         idx = TP_VALUES.index(param_val) if param_val in TP_VALUES else 0
#         n = len(TP_VALUES)
#         frac = idx / (n - 1) if n > 1 else 0
#     else:
#         return (size_min + size_max) / 2
#     return size_min + frac * (size_max - size_min)
# _CONFIG_SWITCH_STYLE = {"sf": ("o", "black"), "bw": ("s", "black"), "tp": ("^", "black")}

# Similar symbols per param (current)
_CONFIG_SWITCH_MARKERS = {
    "sf": ["o", "8", "p", "h", "H", "D"],
    "bw": ["s", "D", "p", "h"],
    "bw_tp_plot": ["^", "v", "<", ">"],
    "tp": ["^", "v", "<"],
}


def _letter_label_from_param(param_type, param_val):
    """Letter labels: BW->B0-B3; SF->S0-S5; TP->T0-T2 (index-based)."""
    if param_val is None:
        return ""
    if param_type == "sf" and param_val in SF_VALUES:
        return f"S{SF_VALUES.index(param_val)}"
    if param_type == "bw" and param_val in BW_VALUES:
        return f"B{BW_VALUES.index(param_val)}"
    if param_type == "tp" and param_val in TP_VALUES:
        return f"T{TP_VALUES.index(param_val)}"
    return ""


def _marker_from_param(param_type, param_val, plot_param=None):
    """Return marker for param value. On TP plot, BW uses triangles to avoid overlap with SF."""
    if param_val is None:
        return "o"
    if param_type == "bw" and plot_param == "tp":
        markers = _CONFIG_SWITCH_MARKERS["bw_tp_plot"]
    else:
        markers = _CONFIG_SWITCH_MARKERS.get(param_type, ["o"])
    if param_type == "sf" and param_val in SF_VALUES:
        idx = SF_VALUES.index(param_val)
    elif param_type == "bw" and param_val in BW_VALUES:
        idx = BW_VALUES.index(param_val)
    elif param_type == "tp" and param_val in TP_VALUES:
        idx = TP_VALUES.index(param_val)
    else:
        return markers[0]
    return markers[min(idx, len(markers) - 1)]


# One marker per param type (combined plot: no value distinction)
_PARAM_MARKERS = {"sf": "o", "bw": "s", "tp": "^"}


def _draw_config_change_markers(ax, points, t_min, t_max, e_min, e_max, plot_param=None, include_params=None, use_markers=False, partner_map=None):
    """Draw letter labels or markers. Dashed lines: partner_map or combined_sequence logic."""
    include_params = include_params or []
    # Build list of (t, e, param_type, param_val, sf, bw, tp) for points in view, sorted by time
    visible = []
    for item in points:
        t, e = item[0], item[1]
        if t_min <= t <= t_max and e_min <= e <= e_max and len(item) > 6:
            visible.append((t, e, item[2], item[3], item[4], item[5], item[6]))
    visible.sort(key=lambda x: x[0])
    # Combined plot: config order SF->BW->TP. Per cycle: SF (spawns with BW1), TP TP, BW TP TP, BW TP TP, BW TP TP
    use_combined_sequence = partner_map == "combined"
    if use_combined_sequence:
        partner_map = None
    elif partner_map is None:
        partner_map = {"sf": [], "bw": ["sf"], "tp": ["bw", "sf"]}
    e_range = e_max - e_min if e_max > e_min else 1
    if use_combined_sequence:
        first_sf_idx = next((i for i, x in enumerate(visible) if x[2] == "sf"), None)
        visible_combined = visible[first_sf_idx:] if first_sf_idx is not None else []
    else:
        visible_combined = visible
    SUB_FONTSIZE = 5
    if use_markers:
        for t, e, param_type, param_val, sf, bw, tp in visible_combined:
            marker = _marker_from_param(param_type, param_val, plot_param)
            ax.scatter([t], [e], marker=marker, s=45, facecolors="white", edgecolors="black",
                       linewidths=1.0, zorder=5)
        if use_combined_sequence:
            t_range = t_max - t_min if t_max > t_min else 1
            sub_size = 18
            sub_offset_x = 0.035 * t_range
            sub_spacing_x = 0.018 * t_range
            for t, e, param_type, param_val, sf, bw, tp in visible_combined:
                if param_type == "sf" and bw is not None and tp is not None:
                    others = [("bw", bw), ("tp", tp)]
                elif param_type == "bw" and tp is not None:
                    others = [("tp", tp)]
                else:
                    others = []
                for i, (p, val) in enumerate(others):
                    marker = _marker_from_param(p, val, "combined")
                    dx = sub_offset_x + i * sub_spacing_x
                    ax.scatter([t + dx], [e], marker=marker, s=sub_size,
                              facecolors="white", edgecolors="black", linewidths=0.8, zorder=5)
        else:
            t_range = t_max - t_min if t_max > t_min else 1
            sub_size = 18
            sub_offset_x = 0.035 * t_range
            sub_spacing_x = 0.018 * t_range
            for t, e, param_type, param_val, sf, bw, tp in visible_combined:
                if param_type == "sf":
                    other_params = [p for p in include_params if p != "sf"]
                elif param_type == "bw":
                    other_params = ["tp"] if "tp" in include_params else []
                else:
                    other_params = []
                if other_params and sf is not None and bw is not None and tp is not None:
                    for i, p in enumerate(other_params):
                        val = sf if p == "sf" else (bw if p == "bw" else tp)
                        if val is None:
                            continue
                        marker = _marker_from_param(p, val, plot_param)
                        dx = sub_offset_x + i * sub_spacing_x
                        ax.scatter([t + dx], [e], marker=marker, s=sub_size,
                                  facecolors="white", edgecolors="black", linewidths=0.8, zorder=5)
    else:
        for i, (t, e, param_type, param_val, sf, bw, tp) in enumerate(visible_combined):
            label = _letter_label_from_param(param_type, param_val)
            if label:
                txt = ax.annotate(label, (t, e), ha="center", va="center", fontsize=MARKER_FONTSIZE,
                                  color="white", zorder=5)
                txt.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground="black")])
            if param_type == "sf":
                other_params = [p for p in include_params if p != "sf"]
            elif param_type == "bw":
                other_params = ["tp"] if "tp" in include_params else []
            else:
                other_params = []
            if other_params and sf is not None and bw is not None and tp is not None:
                other_labels = []
                for p in other_params:
                    val = sf if p == "sf" else (bw if p == "bw" else tp)
                    lbl = _letter_label_from_param(p, val)
                    if lbl:
                        other_labels.append(lbl)
                if other_labels:
                    offset_y = -0.02 * e_range
                    sub = ax.annotate(", ".join(other_labels), (t, e + offset_y), ha="center", va="top",
                                      fontsize=MARKER_FONTSIZE, color="white", zorder=5)
                    sub.set_path_effects([patheffects.withStroke(linewidth=1, foreground="black")])
    if use_combined_sequence:
        # Per config_change_T_init: SF spawns with BW1 -> SF connects to next 2 TPs + next 3 BWs
        # Each BW connects to its next 2 TPs. TP -> empty.
        n_tp_per_bw = len(TP_VALUES) - 1
        n_bw_after_first = 3
        for i, (t, e, param_type, param_val, sf, bw, tp) in enumerate(visible_combined):
            if param_type == "sf":
                tp_count, bw_count = 0, 0
                for j in range(i + 1, len(visible_combined)):
                    pt_t, pt_e, pt_type = visible_combined[j][0], visible_combined[j][1], visible_combined[j][2]
                    if pt_type == "tp" and tp_count < n_tp_per_bw:
                        ax.plot([t, pt_t], [e, pt_e], "k--", linewidth=0.8, zorder=4)
                        tp_count += 1
                    elif pt_type == "bw":
                        if bw_count < n_bw_after_first:
                            ax.plot([t, pt_t], [e, pt_e], "k--", linewidth=0.8, zorder=4)
                            bw_count += 1
                        if bw_count >= n_bw_after_first:
                            break
            elif param_type == "bw":
                tp_count = 0
                for j in range(i + 1, len(visible_combined)):
                    if tp_count >= n_tp_per_bw:
                        break
                    pt_t, pt_e, pt_type = visible_combined[j][0], visible_combined[j][1], visible_combined[j][2]
                    if pt_type == "tp":
                        ax.plot([t, pt_t], [e, pt_e], "k--", linewidth=0.8, zorder=4)
                        tp_count += 1
                    elif pt_type == "bw":
                        break
    else:
        for i, (t, e, param_type, param_val, sf, bw, tp) in enumerate(visible):
            partner_types = partner_map.get(param_type, [])
            if sf is not None and bw is not None and tp is not None:
                for partner_type in partner_types:
                    if partner_type not in include_params:
                        continue
                    partner_val = sf if partner_type == "sf" else (bw if partner_type == "bw" else tp)
                    target_t, target_e = None, None
                    for j in range(i - 1, -1, -1):
                        pt_t, pt_e, pt_type, pt_val, pt_sf, pt_bw, pt_tp = visible[j]
                        if pt_type == partner_type and pt_val == partner_val:
                            target_t, target_e = pt_t, pt_e
                            break
                    if target_t is not None:
                        ax.plot([t, target_t], [e, target_e], "k--", linewidth=0.8, zorder=4)


def _legend_values_and_labels(param_type):
    """Return (values, labels) for legend in ascending order."""
    if param_type == "sf":
        return SF_VALUES, [str(v) for v in SF_VALUES]
    if param_type == "bw":
        return BW_VALUES, [f"{v/1000:.1f}".replace(".0", "") for v in BW_VALUES]
    if param_type == "tp":
        return TP_VALUES, [str(v) for v in TP_VALUES]
    return [], []


def _add_config_switch_legend(ax, include_params, plot_param=None, use_markers=False, horizontal=False):
    """Simple legend: mapping for visible params only (vertical layout for SF plot, horizontal for combined)."""
    if not include_params:
        return
    order = ["sf", "bw", "tp"]
    ordered = [p for p in order if p in include_params]
    mapping = {
        "sf": r"SF7$-$12 $\rightarrow$ S0$-$S5",
        "bw": r"BW62.5$-$500 $\rightarrow$ B0$-$B3",
        "tp": r"TP2$-$22 $\rightarrow$ T0$-$T2",
    }
    if use_markers:
        prefix = {"sf": "SF", "bw": "BW", "tp": "TP"}
        handles, labels = [], []
        msize = 5
        for p in ordered:
            vals, lbls = _legend_values_and_labels(p)
            pre = prefix.get(p, "")
            for i, v in enumerate(vals):
                marker = _marker_from_param(p, v, plot_param)
                handles.append(Line2D([0], [0], linestyle="", marker=marker, markersize=msize,
                                     markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.0))
                labels.append(f"{pre}{lbls[i]}" if lbls else f"{pre}{v}")
        if plot_param == "tp" and ordered == ["sf", "bw"]:
            sf_n, bw_n = len(SF_VALUES), len(BW_VALUES)
            h_sf, h_bw = handles[:sf_n], handles[sf_n:]
            l_sf, l_bw = labels[:sf_n], labels[sf_n:]
            col0 = [h_sf[0], h_sf[2], h_sf[4], h_bw[0], h_bw[2]]
            col1 = [h_sf[1], h_sf[3], h_sf[5], h_bw[1], h_bw[3]]
            lbl0 = [l_sf[0], l_sf[2], l_sf[4], l_bw[0], l_bw[2]]
            lbl1 = [l_sf[1], l_sf[3], l_sf[5], l_bw[1], l_bw[3]]
            handles = col0 + col1
            labels = lbl0 + lbl1
        elif plot_param == "bw" and ordered == ["sf", "tp"]:
            sf_n, tp_n = len(SF_VALUES), len(TP_VALUES)
            h_sf, h_tp = handles[:sf_n], handles[sf_n:]
            l_sf, l_tp = labels[:sf_n], labels[sf_n:]
            col0 = [h_sf[0], h_sf[2], h_sf[4], h_tp[0], h_tp[1]]
            col1 = [h_sf[1], h_sf[3], h_sf[5], h_tp[2]]
            lbl0 = [l_sf[0], l_sf[2], l_sf[4], l_tp[0], l_tp[1]]
            lbl1 = [l_sf[1], l_sf[3], l_sf[5], l_tp[2]]
            handles = col0 + col1
            labels = lbl0 + lbl1
    else:
        handles = [Line2D([0], [0], linestyle="", marker="", markersize=0) for _ in ordered]
        labels = [mapping[p] for p in ordered]
    ncol = 3 if horizontal else (2 if plot_param in ("tp", "bw") else (1 if plot_param in ("sf", "combined") else len(ordered)))
    hlen = 0.6 if (use_markers and plot_param in ("tp", "bw", "sf")) else (1.0 if use_markers else 0)
    leg_fontsize = 8 if horizontal else 9
    leg_kw = dict(fontsize=leg_fontsize, handlelength=hlen, handletextpad=0.3, ncol=ncol,
                  borderaxespad=0.1, borderpad=0.2)
    if plot_param in ("tp", "bw"):
        leg_kw["columnspacing"] = 0.5
        leg_kw["handletextpad"] = 0.2
    loc = "upper center" if horizontal else "upper left"
    bbox = (0.5, 0.98) if horizontal else (0, 1)
    ax.legend(handles=handles, labels=labels, loc=loc, bbox_to_anchor=bbox, **leg_kw)


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
    print("Starting energy-vs-time v3 plot generation...")
    # Override for IEEEtran: uniform 10pt text everywhere (legend, axes, ticks, colorbar, etc.)
    plt.rcParams.update({
        "font.size": IEEE_FONTSIZE,
        "axes.labelsize": IEEE_FONTSIZE,
        "axes.titlesize": IEEE_FONTSIZE,
        "xtick.labelsize": IEEE_FONTSIZE,
        "ytick.labelsize": IEEE_FONTSIZE,
        "legend.fontsize": IEEE_FONTSIZE,
        "legend.title_fontsize": IEEE_FONTSIZE,
        "figure.titlesize": IEEE_FONTSIZE,
        "axes.unicode_minus": False,
    })
    config_dir = os.path.join(WORKSPACE, "results", "raw_test_data_plots")
    version = "v3"
    output_dir = os.path.join(config_dir, "energy_vs_time", version)
    os.makedirs(output_dir, exist_ok=True)
    filters = []
    # Viridis colormap (no white; natural end)
    cmap = mcolors.LinearSegmentedColormap.from_list("per_viridis", plt.cm.viridis_r(np.linspace(0.1, 1, 256)), N=256)
    cmap.set_bad(color=BG_DARK)
    n_time_bins = 30
    n_energy_bins = 30

    pts = collect_all_packets_time_energy_lost(DATA_ROOT, filters)
    if not pts:
        print("No data found.")
        return
    print(f"Collected {len(pts):,} packet points.")

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
    print("Built combined heatmap matrix.")

    plot_combined_heatmap(output_dir, config_dir, DATA_ROOT, filters, time_bins, energy_bins, mat, t_min, t_max, e_min, e_max, vmin, vmax, log_vmin, cmap, suffix=f"_{version}", pts=pts)

    # Focused plot skipped for now

    # Param transitions
    plot_param_transitions_bw(
        output_dir, config_dir, DATA_ROOT, filters, time_bins, energy_bins,
        t_min, t_max, e_min, e_max, vmin, vmax, log_vmin, build_heatmap_matrix_per_packet,
        mat_combined=mat, output_suffix=f"_{version}", pts=pts,
    )
    plot_param_transitions_sf(
        output_dir, config_dir, DATA_ROOT, filters, time_bins, energy_bins,
        t_min, t_max, e_min, e_max, vmin, vmax, log_vmin, build_heatmap_matrix_per_packet,
        mat_combined=mat, output_suffix=f"_{version}", pts=pts,
    )
    plot_param_transitions_tp(
        output_dir, config_dir, DATA_ROOT, filters, time_bins, energy_bins,
        t_min, t_max, e_min, e_max, vmin, vmax, log_vmin, build_heatmap_matrix_per_packet,
        mat_combined=mat, output_suffix=f"_{version}", pts=pts,
    )


def _truncate_cmap_no_white(cmap, minval=0.25):
    """Truncate colormap so 0 (vmin) maps to a visible color, not white."""
    return mcolors.LinearSegmentedColormap.from_list("truncated", cmap(np.linspace(minval, 1, 256)))


def _color_to_black_cmap(base_cmap, name):
    """Create cmap from light color to dark tint so configs stay distinguishable."""
    light = base_cmap(0.5)   # light end of the cmap
    dark = base_cmap(0.95)   # dark tint of same color
    cm = mcolors.LinearSegmentedColormap.from_list(name, [light, dark])
    cm.set_bad(color="none", alpha=0)
    return cm


# Base colormaps for per-value color-to-black (Blues, Reds, Greens, Purples, Oranges, RdPu for SF)
_BASE_CMAPS = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Purples, plt.cm.Oranges, plt.cm.RdPu]


def _legend_color(base_cmap):
    """Characteristic color from cmap (matches color-to-black start)."""
    return base_cmap(0.65)


def _enforce_axis_fontsize(ax):
    """Force axis labels and tick labels to IEEE_FONTSIZE (avoids rcParams override issues)."""
    ax.xaxis.label.set_fontsize(IEEE_FONTSIZE)
    ax.yaxis.label.set_fontsize(IEEE_FONTSIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(IEEE_FONTSIZE)


def _normalize_figure_fonts(fig):
    """Set all text in figure to IEEE_FONTSIZE. Skip intentionally smaller text (markers, annotations)."""
    for obj in fig.findobj(Text):
        if hasattr(obj, "get_fontsize") and obj.get_fontsize() < IEEE_FONTSIZE:
            continue
        obj.set_fontsize(IEEE_FONTSIZE)


def _add_legend_labels(ax, labels, unit=None, pad=0, label_y=None, unit_y=None, label_va="bottom", unit_va="bottom", fontsize=None):
    """Draw rotated labels in the strip below the colorbars."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    n = len(labels)
    if label_y is None:
        label_y = 0.18 if unit else 0.03
    if unit_y is None:
        unit_y = 0.03
    if fontsize is None:
        fontsize = IEEE_FONTSIZE
    if n > 0:
        for i in range(n):
            if pad > 0 and n > 1:
                x = pad + (i / (n - 1)) * (1 - 2 * pad)
            else:
                x = (i + 0.5) / n
            ax.text(
                x,
                label_y,
                labels[i],
                ha="center",
                va=label_va,
                fontsize=fontsize,
                transform=ax.transAxes,
                rotation=90,
                clip_on=False,
            )
    if unit:
        ax.text(
            0.5,
            unit_y,
            f"({unit})",
            ha="center",
            va=unit_va,
            fontsize=fontsize,
            transform=ax.transAxes,
            clip_on=False,
        )


def _tighten_scale_label_gap(fig, n_cbars, gap_reduce=0.05, start_ax=1):
    """Manually move colorbar axes down and legend axes up to reduce gap."""
    cbar_axes = fig.axes[start_ax:start_ax + n_cbars]
    ax_legend = fig.axes[start_ax + n_cbars]
    leg_pos = ax_legend.get_position()
    for cax in cbar_axes:
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 - gap_reduce, pos.width, pos.height + gap_reduce])
    ax_legend.set_position([leg_pos.x0, leg_pos.y0 + gap_reduce, leg_pos.width, leg_pos.height])


def _shift_scale_labels_down(fig, n_cbars, shift=0.06, start_ax=1):
    """Move the whole scale+labels block down."""
    cbar_axes = fig.axes[start_ax:start_ax + n_cbars]
    ax_legend = fig.axes[start_ax + n_cbars]
    for cax in cbar_axes:
        pos = cax.get_position()
        cax.set_position([pos.x0, pos.y0 - shift, pos.width, pos.height])
    leg_pos = ax_legend.get_position()
    ax_legend.set_position([leg_pos.x0, leg_pos.y0 - shift, leg_pos.width, leg_pos.height])


def _pull_scales_closer(fig, n_cbars, shift_left=0.14, start_ax=1):
    """Move colorbar axes left to reduce gap between plot and scales."""
    cbar_axes = fig.axes[start_ax:start_ax + n_cbars]
    ax_legend = fig.axes[start_ax + n_cbars]
    for cax in cbar_axes:
        pos = cax.get_position()
        cax.set_position([pos.x0 - shift_left, pos.y0, pos.width, pos.height])
    leg_pos = ax_legend.get_position()
    ax_legend.set_position([leg_pos.x0 - shift_left, leg_pos.y0, leg_pos.width, leg_pos.height])


def _reserve_bottom_space_for_scale_labels(ax, extra=0.05):
    """Shrink the main plot slightly to free bottom space for rotated scale labels."""
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + extra, pos.width, pos.height - extra])


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
        cbar.outline.set_linewidth(0.9)
        if i < n - 1:
            cbar.ax.set_yticks([])
            cbar.ax.set_yticklabels([])
        else:
            cbar.ax.tick_params(labelsize=IEEE_FONTSIZE)


def _add_labels_inside_colorbars(fig, labels, start_ax=1, y=0.01, fontsize=None, color="black", stroke_color=None):
    """Place one rotated label inside each colorbar axis, centered vertically."""
    if fontsize is None:
        fontsize = IEEE_FONTSIZE
    for i, label in enumerate(labels):
        cax = fig.axes[start_ax + i]
        txt = cax.text(
            0.5,
            y,
            label,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=fontsize,
            color=color,
            transform=cax.transAxes,
            clip_on=True,
        )
        if stroke_color:
            txt.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground=stroke_color)])


def _align_colorbars_to_plot(fig, n_cbars, start_ax=1):
    """Set colorbar axes to match main plot vertical extent (top to bottom of plot window)."""
    ax_main = fig.axes[0]
    main_pos = ax_main.get_position()
    y0, height = main_pos.y0, main_pos.height
    for i in range(n_cbars):
        cax = fig.axes[start_ax + i]
        pos = cax.get_position()
        cax.set_position([pos.x0, y0, pos.width, height])


def _position_scale_label_strip(fig, n_cbars, bottom=0.03, gap=0.01, start_ax=1):
    """Place the label strip directly below the colorbars, fully inside the figure."""
    ax_main = fig.axes[0]
    cbar_axes = fig.axes[start_ax:start_ax + n_cbars]
    ax_legend = fig.axes[start_ax + n_cbars]
    if not cbar_axes:
        return
    main_pos = ax_main.get_position()
    x0 = min(ax.get_position().x0 for ax in cbar_axes)
    x1 = max(ax.get_position().x0 + ax.get_position().width for ax in cbar_axes)
    y0 = bottom
    y1 = max(bottom + 0.04, main_pos.y0 - gap)
    ax_legend.set_position([x0, y0, x1 - x0, y1 - y0])


def _add_per_label_above_colorbars(fig, n_cbars, start_ax=1):
    """Add PER (%) label inside the main plot (upper-right corner)."""
    if not fig.axes:
        return
    ax_main = fig.axes[0]
    ax_main.text(
        0.98,
        0.98,
        "PER (%)",
        ha="right",
        va="top",
        fontsize=IEEE_FONTSIZE,
        color="black",
        transform=ax_main.transAxes,
        zorder=6,
    )



def plot_combined_heatmap(output_dir, config_dir, data_root, filters, time_bins, energy_bins, mat, t_min, t_max, e_min, e_max, vmin, vmax, log_vmin, cmap, suffix="", pts=None):
    """Combined PER heatmap with annotated SF/BW config regions showing performance-energy patterns."""
    print(f"Rendering combined heatmap{suffix}...")
    fig = plt.figure(figsize=FIGSIZE_TWO_COL)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.12], wspace=0)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(BG_DARK)
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.tick_params(axis="both", colors="black", labelsize=IEEE_FONTSIZE)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

    # Draw PER heatmap
    scaled = np.where(np.isnan(mat), np.nan, np.maximum(mat * 0.2, log_vmin))
    mat_masked = np.ma.masked_where(np.isnan(mat), scaled)
    im = ax.pcolormesh(
        time_bins, energy_bins, mat_masked,
        cmap=cmap, norm=LogNorm(vmin=log_vmin, vmax=vmax), shading="flat",
    )
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(e_min, e_max)
    ax.set_aspect((t_max - t_min) / (e_max - e_min))
    ax.set_xlabel(r"$T_{\mathrm{init}}$ (min)", fontsize=IEEE_FONTSIZE)
    ax.set_ylabel("Energy (mJ)", fontsize=IEEE_FONTSIZE)
    _enforce_axis_fontsize(ax)

    # --- Config region annotations ---
    switches = _load_config_switches_from_csv(config_dir)
    if switches:
        # Group configs by SF
        sf_configs = {}
        for tm, sf, bw, tp in switches:
            sf_configs.setdefault(sf, []).append((tm, bw, tp))
        sorted_sfs = sorted(sf_configs.keys())

        # Build SF regions: (sf, t_start_min, t_end_min)
        sf_regions = []
        for i, sf in enumerate(sorted_sfs):
            ts = min(t for t, _, _ in sf_configs[sf])
            if i + 1 < len(sorted_sfs):
                te = min(t for t, _, _ in sf_configs[sorted_sfs[i + 1]])
            else:
                times_sf = sorted(t for t, _, _ in sf_configs[sf])
                te = times_sf[-1] + (times_sf[-1] - times_sf[-2]) if len(times_sf) >= 2 else t_max
            sf_regions.append((sf, ts, min(te, t_max)))

        e_range = e_max - e_min

        # Draw SF region bands with characteristic colors
        for idx, (sf, ts, te) in enumerate(sf_regions):
            ts_c = max(ts, t_min)
            te_c = min(te, t_max)
            if ts_c >= te_c:
                continue
            band_color = _BASE_CMAPS[idx % len(_BASE_CMAPS)](0.18)
            ax.axvspan(ts_c, te_c, color=band_color, alpha=0.35, zorder=0.5)

            # SF transition line (solid, between regions)
            if idx > 0 and ts >= t_min:
                ax.axvline(ts, color="black", linewidth=0.8, alpha=0.5, zorder=2)

            # SF label vertical inside upper-left of band (lowered)

            
            
            if idx < 3:
                x_label = ts_c + (t_max - t_min) * 0.012
                if idx < 1:
                    x_label = ts_c + (t_max - t_min) * 0.012
                y_label = e_max - e_range * 0.2
            else:
                x_label = ts_c + (t_max - t_min) * 0.012
                y_label = e_max - e_range * 0.09
            txt = ax.text(x_label, y_label, f"SF{sf}", ha="right", va="top",
                            fontsize=IEEE_FONTSIZE, color="black", zorder=6,
                            rotation=90, rotation_mode="anchor")
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground="white")])

            # Per-SF statistics from raw packets
            if pts:
                sf_pts = [p for p in pts if ts <= p[0] < te]
                if sf_pts:
                    n_total = len(sf_pts)
                    n_lost = sum(1 for p in sf_pts if p[2] == 1)
                    per_val = 100.0 * n_lost / n_total
                    avg_e = float(np.mean([p[1] for p in sf_pts]))

                    # PER % — vertical for first two SFs, horizontal for the rest
                    y_per = e_max - e_range*0.015
                    if idx < 3:
                        sub = ax.text(x_label, y_per - e_range*0.15, f"{per_val:.1f}%", ha="left", va="top",
                                     fontsize=IEEE_FONTSIZE, color="black", zorder=6,
                                     rotation=90, rotation_mode="anchor")
                    else:
                        sub = ax.text(x_label, y_per, f"{per_val:.1f}%", ha="left", va="top",
                                     fontsize=IEEE_FONTSIZE, color="black", zorder=6,
                                     rotation=0, rotation_mode="anchor")
                    sub.set_path_effects([patheffects.withStroke(linewidth=2, foreground="white")])

        # Configuration path: use switch times and energy averaged across all distances (v2 style)
        if pts:
            path_times = []
            path_energies = []
            path_configs = []
            for i, (tm, sf, bw, tp) in enumerate(switches):
                avg_e = _config_avg_energy(data_root, filters, sf, bw, tp)
                if avg_e is not None:
                    path_times.append(tm)
                    path_energies.append(avg_e)
                    path_configs.append((sf, bw, tp))
            if len(path_times) >= 2:
                ax.plot(path_times, path_energies, color="white", linewidth=2.0, zorder=3, alpha=0.9)
                ax.plot(path_times, path_energies, color="black", linewidth=1.0, zorder=3.5, alpha=0.9, label="cfg path")
                ax.legend(loc="lower right", fontsize=IEEE_FONTSIZE - 1, framealpha=0.8, edgecolor="none",
                          borderpad=0.2, handlelength=1.0, handletextpad=0.3, borderaxespad=0.2)

    # Compact colorbar
    cax = fig.add_subplot(gs[1])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=IEEE_FONTSIZE)
    _align_colorbars_to_plot(fig, 1)
    # Pull colorbar closer to plot (manual shift — combined plot has no legend axis)
    cax_pos = cax.get_position()
    cax.set_position([cax_pos.x0 - 0.15, cax_pos.y0, cax_pos.width, cax_pos.height])

    # PER label below the colorbar, aligned with x-axis tick values
    ax_pos = ax.get_position()
    cax_pos2 = cax.get_position()
    # Align with x-axis tick labels (at plot bottom)
    y_xlabel = ax_pos.y0 - 0.02
    fig.text(cax_pos2.x0 + cax_pos2.width / 2, y_xlabel, "PER (%)",
             ha="center", va="top", fontsize=IEEE_FONTSIZE, color="black")

    _normalize_figure_fonts(fig)
    fname = f"raw_per_gradient_energy_vs_time{suffix}.png" if suffix else "raw_per_gradient_energy_vs_time.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_param_transitions_bw(output_dir, config_dir, data_root, filters, time_bins, energy_bins, t_min, t_max, e_min, e_max, vmin, vmax, log_vmin, build_matrix_fn, mat_combined=None, output_suffix="", pts=None):
    """BW transitions: one color-to-black cmap per BW, annotated BW region bands with SF separators."""
    print(f"Rendering BW transition heatmap{output_suffix}...")
    fig = plt.figure(figsize=FIGSIZE_TWO_COL)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.12], wspace=0.0014)
    ax = fig.add_subplot(gs[0])
    _reserve_bottom_space_for_scale_labels(ax)
    ax.set_facecolor(BG_DARK)
    ax.tick_params(axis="both", colors="black", labelsize=IEEE_FONTSIZE)
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
        ax.pcolormesh(
            time_bins, energy_bins, mat_masked,
            cmap=cmap, norm=LogNorm(vmin=log_vmin, vmax=vmax), shading="flat",
        )
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(e_min, e_max)
    ax.set_aspect((t_max - t_min) / (e_max - e_min))
    ax.set_xlabel(r"$T_{\mathrm{init}}$ (min)", fontsize=IEEE_FONTSIZE)
    ax.set_ylabel("Energy (mJ)", fontsize=IEEE_FONTSIZE)
    _enforce_axis_fontsize(ax)

    # Annotated BW region bands
    switches = _load_config_switches_from_csv(config_dir)
    if switches:
        bw_configs = {}
        for tm, sf, bw, tp in switches:
            bw_configs.setdefault(bw, []).append((tm, sf, tp))
        sorted_bws = sorted(bw_configs.keys())
        e_range = e_max - e_min

        # BW regions: within each SF, BW configs appear in order; group by BW across all SFs
        # Show per-BW statistics
        for idx, bw in enumerate(sorted_bws):
            bw_label = f"BW{bw/1000:.1f}".replace(".0", "")
            band_color = _BASE_CMAPS[idx % len(_BASE_CMAPS)](0.25)
            if pts:
                bw_pts = [p for p in collect_all_packets_time_energy_lost(data_root, filters, include_bw=bw)]
                if bw_pts:
                    n_total = len(bw_pts)
                    n_lost = sum(1 for p in bw_pts if p[2] == 1)
                    per_val = 100.0 * n_lost / n_total
                    avg_e = float(np.mean([p[1] for p in bw_pts]))

        # SF separation lines
        sf_configs = {}
        for tm, sf, bw, tp in switches:
            sf_configs.setdefault(sf, []).append((tm, bw, tp))
        sorted_sfs = sorted(sf_configs.keys())
        for i, sf in enumerate(sorted_sfs):
            ts = min(t for t, _, _ in sf_configs[sf])
            if i + 1 < len(sorted_sfs):
                te = min(t for t, _, _ in sf_configs[sorted_sfs[i + 1]])
            else:
                times_sf = sorted(t for t, _, _ in sf_configs[sf])
                te = times_sf[-1] + (times_sf[-1] - times_sf[-2]) if len(times_sf) >= 2 else t_max
            ts_c, te_c = max(ts, t_min), min(te, t_max)
            if ts_c >= te_c:
                continue
            # SF separator line
            if i > 0 and ts >= t_min:
                ax.axvline(ts, color="black", linewidth=0.8, alpha=0.5, zorder=2)
            # SF label vertical inside upper-left of band (lowered)
            x_label = ts_c + (t_max - t_min) * 0.01
            y_label = e_max - e_range * 0.01
            txt = ax.text(x_label, y_label, f"SF{sf}", ha="right", va="top",
                         fontsize=IEEE_FONTSIZE, color="black", zorder=6,
                         rotation=90, rotation_mode="anchor")
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground="white")])

        # Configuration path: one point per (SF, BW, TP) in actual measurement order
        # Follows SF -> BW1(TP1,2,3) -> BW2(TP1,2,3) -> BW3(TP1,2,3) -> SF8 -> ...
        if pts:
            path_times = []
            path_energies = []
            path_labels = []
            for tm, sf, bw, tp in switches:
                avg_e = _config_avg_energy(data_root, filters, sf, bw, tp)
                if avg_e is not None:
                    path_times.append(tm)
                    path_energies.append(avg_e)
                    path_labels.append(f"SF{sf}_BW{bw}_TP{tp}")
            if len(path_times) >= 2:
                ax.plot(path_times, path_energies, color="white", linewidth=1.5, zorder=3, alpha=0.9)
                ax.plot(path_times, path_energies, color="black", linewidth=0.8, zorder=3.5, alpha=0.9, label="cfg path")
                ax.legend(loc="lower right", fontsize=IEEE_FONTSIZE - 1, framealpha=0.8, edgecolor="none",
                          borderpad=0.2, handlelength=1.0, handletextpad=0.3, borderaxespad=0.2)

    labels_bw = [f"BW{bw/1000:.1f}".replace(".0", "") for bw in BW_VALUES]
    gs_right = gs[1].subgridspec(2, 1, height_ratios=[0.64, 0.36], hspace=0)
    gs_cbars = gs_right[0].subgridspec(1, len(BW_VALUES), wspace=0.02)
    _add_param_colorbars(fig, gs_cbars, BW_VALUES, log_vmin, vmax)
    ax_leg = fig.add_subplot(gs_right[1])
    _add_labels_inside_colorbars(fig, labels_bw, color="black")
    _add_legend_labels(ax_leg, [], unit="kHz", unit_y=0.88, unit_va="top")
    _pull_scales_closer(fig, len(BW_VALUES))
    _align_colorbars_to_plot(fig, len(BW_VALUES))
    _position_scale_label_strip(fig, len(BW_VALUES), bottom=0.02, gap=0.008)
    _normalize_figure_fonts(fig)
    fig.savefig(os.path.join(output_dir, f"raw_per_gradient_energy_vs_time_bw_transitions{output_suffix}.png"), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: raw_per_gradient_energy_vs_time_bw_transitions{output_suffix}.png")


def plot_param_transitions_sf(output_dir, config_dir, data_root, filters, time_bins, energy_bins, t_min, t_max, e_min, e_max, vmin, vmax, log_vmin, build_matrix_fn, mat_combined=None, output_suffix="", pts=None):
    """SF transitions: one color-to-black cmap per SF, annotated SF region bands with BW separators and PER stats."""
    print(f"Rendering SF transition heatmap{output_suffix}...")
    fig = plt.figure(figsize=FIGSIZE_TWO_COL)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.12], wspace=0.0014)
    ax = fig.add_subplot(gs[0])
    _reserve_bottom_space_for_scale_labels(ax)
    ax.set_facecolor(BG_DARK)
    ax.tick_params(axis="both", colors="black", labelsize=IEEE_FONTSIZE)
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
        ax.pcolormesh(
            time_bins, energy_bins, mat_masked,
            cmap=cmap, norm=LogNorm(vmin=log_vmin, vmax=vmax), shading="flat",
        )
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(e_min, e_max)
    ax.set_aspect((t_max - t_min) / (e_max - e_min))
    ax.set_xlabel(r"$T_{\mathrm{init}}$ (min)", fontsize=IEEE_FONTSIZE)
    ax.set_ylabel("Energy (mJ)", fontsize=IEEE_FONTSIZE)
    _enforce_axis_fontsize(ax)

    # Annotated SF region bands with BW separators
    switches = _load_config_switches_from_csv(config_dir)
    if switches:
        sf_configs = {}
        for tm, sf, bw, tp in switches:
            sf_configs.setdefault(sf, []).append((tm, bw, tp))
        sorted_sfs = sorted(sf_configs.keys())
        e_range = e_max - e_min

        for i, sf in enumerate(sorted_sfs):
            ts = min(t for t, _, _ in sf_configs[sf])
            if i + 1 < len(sorted_sfs):
                te = min(t for t, _, _ in sf_configs[sorted_sfs[i + 1]])
            else:
                times_sf = sorted(t for t, _, _ in sf_configs[sf])
                te = times_sf[-1] + (times_sf[-1] - times_sf[-2]) if len(times_sf) >= 2 else t_max
            ts_c, te_c = max(ts, t_min), min(te, t_max)
            if ts_c >= te_c:
                continue
            band_color = _BASE_CMAPS[i % len(_BASE_CMAPS)](0.18)
            ax.axvspan(ts_c, te_c, color=band_color, alpha=0.25, zorder=0.5)

            if i > 0 and ts >= t_min:
                ax.axvline(ts, color="black", linewidth=0.8, alpha=0.5, zorder=2)

            # SF label vertical inside upper-left of band (lowered)
            x_label = ts_c + (t_max - t_min) * 0.01
            y_label = e_max - e_range * 0.01
            txt = ax.text(x_label, y_label, f"SF{sf}", ha="right", va="top",
                         fontsize=IEEE_FONTSIZE, color="black", zorder=6,
                         rotation=90, rotation_mode="anchor")
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground="white")])

            if pts:
                sf_pts = [p for p in pts if ts <= p[0] < te]

        # Configuration path: SF→BW→TP order (within each SF: BW62.5-TP2,12,22 → BW125-TP2,12,22 → ... → BW500-TP2,12,22)
        if pts:
            # Store each config's switch time and energy (v2 style: switch time + _config_avg_energy)
            config_data = {}  # (sf, bw, tp) -> (switch_time, avg_energy)
            for i, (tm, sf, bw, tp) in enumerate(switches):
                key = (sf, bw, tp)
                avg_e = _config_avg_energy(data_root, filters, sf, bw, tp)
                if avg_e is not None:
                    config_data[key] = (tm, avg_e)
            # Iterate SF→BW→TP: each SF in sequence, within SF go BW62.5→500 with all TPs
            path_times = []
            path_energies = []
            path_labels = []
            for sf in SF_VALUES:
                for bw in BW_VALUES:
                    for tp in TP_VALUES:
                        key = (sf, bw, tp)
                        if key in config_data:
                            tm, avg_e = config_data[key]
                            path_times.append(tm)
                            path_energies.append(avg_e)
                            path_labels.append(f"SF{sf}_BW{bw}_TP{tp}")
            if len(path_times) >= 2:
                ax.plot(path_times, path_energies, color="white", linewidth=1.5, zorder=3, alpha=0.9)
                ax.plot(path_times, path_energies, color="black", linewidth=0.8, zorder=3.5, alpha=0.9, label="cfg path")
                ax.legend(loc="lower right", fontsize=IEEE_FONTSIZE - 1, framealpha=0.8, edgecolor="none",
                          borderpad=0.2, handlelength=1.0, handletextpad=0.3, borderaxespad=0.2)

    labels_sf = [f"SF{sf}" for sf in SF_VALUES]
    gs_right = gs[1].subgridspec(2, 1, height_ratios=[0.64, 0.36], hspace=0)
    gs_cbars = gs_right[0].subgridspec(1, len(SF_VALUES), wspace=0.03)
    _add_param_colorbars(fig, gs_cbars, SF_VALUES, log_vmin, vmax)
    ax_leg = fig.add_subplot(gs_right[1])
    _add_legend_labels(ax_leg, labels_sf, pad=0.02, label_y=0.98, label_va="top")
    _pull_scales_closer(fig, len(SF_VALUES))
    _align_colorbars_to_plot(fig, len(SF_VALUES))
    _position_scale_label_strip(fig, len(SF_VALUES), bottom=0.02, gap=0.008)
    _normalize_figure_fonts(fig)
    fig.savefig(os.path.join(output_dir, f"raw_per_gradient_energy_vs_time_sf_transitions{output_suffix}.png"), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: raw_per_gradient_energy_vs_time_sf_transitions{output_suffix}.png")


def plot_param_transitions_tp(output_dir, config_dir, data_root, filters, time_bins, energy_bins, t_min, t_max, e_min, e_max, vmin, vmax, log_vmin, build_matrix_fn, mat_combined=None, output_suffix="", pts=None):
    """TP transitions: one color-to-black cmap per TP, annotated SF region bands with BW separators."""
    print(f"Rendering TP transition heatmap{output_suffix}...")
    fig = plt.figure(figsize=FIGSIZE_TWO_COL)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 0.12], wspace=0.0014)
    ax = fig.add_subplot(gs[0])
    _reserve_bottom_space_for_scale_labels(ax)
    ax.set_facecolor(BG_DARK)
    ax.tick_params(axis="both", colors="black", labelsize=IEEE_FONTSIZE)
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
        ax.pcolormesh(
            time_bins, energy_bins, mat_masked,
            cmap=cmap, norm=LogNorm(vmin=log_vmin, vmax=vmax), shading="flat",
        )
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(e_min, e_max)
    ax.set_aspect((t_max - t_min) / (e_max - e_min))
    ax.set_xlabel(r"$T_{\mathrm{init}}$ (min)", fontsize=IEEE_FONTSIZE)
    ax.set_ylabel("Energy (mJ)", fontsize=IEEE_FONTSIZE)
    _enforce_axis_fontsize(ax)

    # Annotated SF region bands with BW separators
    switches = _load_config_switches_from_csv(config_dir)
    if switches:
        sf_configs = {}
        for tm, sf, bw, tp in switches:
            sf_configs.setdefault(sf, []).append((tm, bw, tp))
        sorted_sfs = sorted(sf_configs.keys())
        e_range = e_max - e_min

        for i, sf in enumerate(sorted_sfs):
            ts = min(t for t, _, _ in sf_configs[sf])
            if i + 1 < len(sorted_sfs):
                te = min(t for t, _, _ in sf_configs[sorted_sfs[i + 1]])
            else:
                times_sf = sorted(t for t, _, _ in sf_configs[sf])
                te = times_sf[-1] + (times_sf[-1] - times_sf[-2]) if len(times_sf) >= 2 else t_max
            ts_c, te_c = max(ts, t_min), min(te, t_max)
            if ts_c >= te_c:
                continue

            if i > 0 and ts >= t_min:
                ax.axvline(ts, color="black", linewidth=0.8, alpha=0.5, zorder=2)

            # SF label vertical inside upper-left of band (lowered)
            x_label = ts_c + (t_max - t_min) * 0.01
            y_label = e_max - e_range * 0.01
            txt = ax.text(x_label, y_label, f"SF{sf}", ha="right", va="top",
                         fontsize=IEEE_FONTSIZE, color="black", zorder=6,
                         rotation=90, rotation_mode="anchor")
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground="white")])

        # Configuration path: simplified (one point per SF,BW pair)
        if pts:
            # Group configs by (sf, bw) - use switch time and _config_avg_energy (v2 style)
            sf_bw_data = {}  # (sf, bw) -> (first_switch_time, [energies])
            for i, (tm, sf, bw, tp) in enumerate(switches):
                key = (sf, bw)
                avg_e = _config_avg_energy(data_root, filters, sf, bw, tp)
                if avg_e is not None:
                    if key not in sf_bw_data:
                        sf_bw_data[key] = (tm, [])
                    sf_bw_data[key][1].append(avg_e)
            # Build path in order: SF7->all BWs, SF8->all BWs, etc.
            path_times = []
            path_energies = []
            for sf in SF_VALUES:
                for bw in BW_VALUES:
                    key = (sf, bw)
                    if key in sf_bw_data:
                        first_tm, energies = sf_bw_data[key]
                        path_times.append(first_tm)
                        path_energies.append(float(np.mean(energies)))
            if len(path_times) >= 2:
                ax.plot(path_times, path_energies, color="white", linewidth=1.5, zorder=3, alpha=0.9)
                ax.plot(path_times, path_energies, color="black", linewidth=0.8, zorder=3.5, alpha=0.9, label="cfg path")
                ax.legend(loc="lower right", fontsize=IEEE_FONTSIZE - 1, framealpha=0.8, edgecolor="none",
                          borderpad=0.2, handlelength=1.0, handletextpad=0.3, borderaxespad=0.2)

    labels_tp = [f"TP{tp}" for tp in TP_VALUES]
    gs_right = gs[1].subgridspec(2, 1, height_ratios=[0.64, 0.36], hspace=0)
    gs_cbars = gs_right[0].subgridspec(1, len(TP_VALUES), wspace=0.02)
    _add_param_colorbars(fig, gs_cbars, TP_VALUES, log_vmin, vmax)
    ax_leg = fig.add_subplot(gs_right[1])
    _add_labels_inside_colorbars(fig, labels_tp, color="black")
    _add_legend_labels(ax_leg, [], unit="dBm", unit_y=0.88, unit_va="top")
    _pull_scales_closer(fig, len(TP_VALUES))
    _align_colorbars_to_plot(fig, len(TP_VALUES))
    _position_scale_label_strip(fig, len(TP_VALUES), bottom=0.02, gap=0.008)
    _normalize_figure_fonts(fig)
    fig.savefig(os.path.join(output_dir, f"raw_per_gradient_energy_vs_time_tp_transitions{output_suffix}.png"), dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: raw_per_gradient_energy_vs_time_tp_transitions{output_suffix}.png")


if __name__ == "__main__":
    main()
