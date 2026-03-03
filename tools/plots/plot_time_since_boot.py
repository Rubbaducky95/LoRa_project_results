"""
Plot time_since_transmission_init_ms across packets.
One subplot per distance: packet index (x) vs time_since_transmission_init_ms (y).
- Solid smooth curve: time_since_transmission_init_ms
- Dotted lines: original time_since_boot inconsistencies from time_reset_locations.csv
"""
import csv
import os
import re
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_config import IEEE_FONTSIZE, FIGSIZE_TWO_COL, SAVE_DPI


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
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def gaussian_smooth(y, sigma=5):
    """Simple Gaussian smoothing using convolution."""
    n = len(y)
    if n < 3 or sigma < 0.5:
        return y
    k = int(min(sigma * 3, n // 2))
    x = np.arange(-k, k + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(y.astype(float), k, mode="edge")
    return np.convolve(padded, kernel, mode="valid")

WORKSPACE = r"C:\Users\ruben\Documents\LoRa_project_results"
DATA_ROOT = os.path.join(WORKSPACE, "raw_test_data")
RESET_CSV = os.path.join(WORKSPACE, "results", "time_reset_locations.csv")

BW_VALUES = [62500, 125000, 250000, 500000]
TP_VALUES = [2, 12, 22]
SF_VALUES = [7, 8, 9, 10, 11, 12]


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    return tuple(map(int, m.groups())) if m else None


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def get_file_order():
    for sf in SF_VALUES:
        for bw in BW_VALUES:
            for tp in TP_VALUES:
                yield (sf, bw, tp)


def format_sf_bw_label(raw_label):
    """Convert 'SF7 BW62.5' or 'SF7_BW62.5' to '(7, 62.5)' (numbers only)."""
    s = raw_label.replace("_", " ")
    m = re.match(r"SF(\d+)\s*BW([\d.]+)", s)
    if m:
        return f"({m.group(1)}, {m.group(2)})"
    return s


def e3_format(x, pos):
    """Format y-axis as Xe3 (e.g. 3.5e3, 35e3)."""
    if x == 0:
        return "0"
    m = x / 1000
    return f"{m:.1f}e3" if m != int(m) else f"{int(m)}e3"


def find_file_in_dist(dist_path, sf, bw, tp):
    for sub in os.listdir(dist_path):
        spath = os.path.join(dist_path, sub)
        if not os.path.isdir(spath):
            continue
        for fn in os.listdir(spath):
            cfg = parse_cfg(fn)
            if cfg and cfg == (sf, bw, tp):
                return os.path.join(spath, fn)
    return None


def load_reset_locations():
    """Return {distance: [(packet_idx, prev_ms, curr_ms), ...]}."""
    by_dist = {}
    if not os.path.isfile(RESET_CSV):
        return by_dist
    with open(RESET_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("distance") and row.get("packet_global_idx"):
                try:
                    dist = row["distance"]
                    idx = int(row["packet_global_idx"])
                    prev = float(row["prev_time_ms"])
                    curr = float(row["curr_time_ms"])
                    by_dist.setdefault(dist, []).append((idx, prev, curr))
                except (ValueError, KeyError):
                    continue
    return by_dist


def collect_times_for_distance(dpath, time_col="time_since_transmission_init_ms"):
    """Return (packet_indices, times_list, config_ids) for a distance folder."""
    indices = []
    times = []
    config_ids = []
    idx = 0
    for sf, bw, tp in get_file_order():
        path = find_file_in_dist(dpath, sf, bw, tp)
        if not path or not os.path.isfile(path):
            continue
        rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
        if not rows:
            continue
        header = rows[0]
        if time_col not in header:
            continue
        time_idx = header.index(time_col)
        payload_idx = header.index("payload") if "payload" in header else -1
        cfg_id = f"SF{sf}_BW{bw//1000}_TP{tp}"
        for r in rows[1:]:
            if payload_idx >= 0 and len(r) > payload_idx and str(r[payload_idx]).strip().startswith("CFG "):
                continue
            if len(r) <= time_idx or not r[time_idx]:
                continue
            try:
                t = float(r[time_idx])
                indices.append(idx)
                times.append(t)
                config_ids.append(cfg_id)
                idx += 1
            except (ValueError, TypeError):
                continue
    return np.array(indices), np.array(times), config_ids


def collect_times_all_distances_overlaid():
    """Return (dist_data, sf_bw_changes_with_labels, all_config_entries).
    dist_data: list of (indices, times) per distance, each with packet index 0,1,2,...
    sf_bw_changes_with_labels: list of (pkt_idx, label, tp) where SF or BW changes.
    all_config_entries: list of (pkt_idx, sf, bw, tp, t_ms) for first packet of each (sf,bw,tp) - for CSV."""
    dist_folders = sorted(
        [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith("distance_")],
        key=lambda x: parse_distance(x) or 0,
    )
    dist_data = []
    sf_bw_changes_with_labels = []
    all_config_entries = []

    for dn in dist_folders:
        dpath = os.path.join(DATA_ROOT, dn)
        if parse_distance(dn) is None:
            continue
        indices = []
        times = []
        prev_sf, prev_bw = None, None
        for sf, bw, tp in get_file_order():
            path = find_file_in_dist(dpath, sf, bw, tp)
            if not path or not os.path.isfile(path):
                continue
            rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
            if not rows:
                continue
            header = rows[0]
            if "time_since_transmission_init_ms" not in header:
                continue
            time_idx = header.index("time_since_transmission_init_ms")
            payload_idx = header.index("payload") if "payload" in header else -1

            first_packet_of_config = True
            for r in rows[1:]:
                if payload_idx >= 0 and len(r) > payload_idx and str(r[payload_idx]).strip().startswith("CFG "):
                    continue
                if len(r) <= time_idx or not r[time_idx]:
                    continue
                try:
                    t = float(r[time_idx])
                    pkt_idx = len(indices)
                    if first_packet_of_config:
                        if len(dist_data) == 0:
                            all_config_entries.append((pkt_idx, sf, bw, tp, t))
                            if prev_sf is not None and (sf != prev_sf or bw != prev_bw):
                                label = f"SF{sf} BW{bw/1000}" if bw % 1000 else f"SF{sf} BW{bw//1000}"
                                sf_bw_changes_with_labels.append((pkt_idx, label, tp))
                        first_packet_of_config = False
                        prev_sf, prev_bw = sf, bw
                    indices.append(pkt_idx)
                    times.append(t)
                except (ValueError, TypeError):
                    continue
        if indices:
            dist_data.append((np.array(indices), np.array(times)))
    return dist_data, sf_bw_changes_with_labels, all_config_entries


def main():
    setup_plot_style()
    reset_by_dist = load_reset_locations()
    dist_folders = sorted(
        [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith("distance_")],
        key=lambda x: parse_distance(x) or 0,
    )
    n = len(dist_folders)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    out_dir = os.path.join(WORKSPACE, "results", "raw_test_data_plots", "cfg_vs_time")
    os.makedirs(out_dir, exist_ok=True)

    # Combined plot v1: averaged across distances, config labels on X-axis, dotted lines cut at curve
    dist_data, sf_bw_changes, all_config_entries = collect_times_all_distances_overlaid()
    if dist_data:
        max_len = max(len(times) for _, times in dist_data)
        times_matrix = np.full((len(dist_data), max_len), np.nan)
        for i, (indices, times) in enumerate(dist_data):
            times_matrix[i, :len(times)] = times
        avg_times = np.nanmean(times_matrix, axis=0)
        valid = ~np.isnan(avg_times)
        indices_avg = np.arange(max_len)[valid]
        avg_times = avg_times[valid]
        sigma = min(5, max(1, len(avg_times) // 100))
        times_smooth = gaussian_smooth(avg_times, sigma=sigma)

        # Combined plot v3: packet index on X-axis, dotted lines from curve up, labels black and rotated
        FONTSIZE = IEEE_FONTSIZE
        dotted_positions = [0] + [p for p, _, _ in sf_bw_changes]
        fig4, ax4 = plt.subplots(figsize=FIGSIZE_TWO_COL)
        ax4.plot(indices_avg, times_smooth / 1000, color="#1f77b4", linewidth=2.2, alpha=0.9)
        ax4.set_xlabel("Packet Index", fontsize=FONTSIZE)
        ax4.set_ylabel(r"$T_{\mathrm{init}}$ (s)", fontsize=FONTSIZE)
        ax4.tick_params(axis="both", labelsize=FONTSIZE)
        xmax = ax4.get_xlim()[1]
        ax4.set_xlim(xmax=xmax - 300)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(e3_format))
        ax4.grid(True, alpha=0.3)
        ymax = ax4.get_ylim()[1]
        ymin = ax4.get_ylim()[0]
        y_range = ymax - ymin
        min_width_cfg = 6
        min_width_ts = 8
        cfg_box_style = dict(
            boxstyle="round,pad=0.06,rounding_size=0.5",
            facecolor="#b8d4e8",
            edgecolor="#2c5aa0",
            linewidth=0.7,
            alpha=0.95,
        )
        ts_box_style = dict(
            boxstyle="round,pad=0.12",
            facecolor="#ffe4b8",
            edgecolor="#b8860b",
            linewidth=0.9,
            alpha=0.95,
        )
        connector_style = dict(color="#5a8ab8", linestyle="--", linewidth=0.8, alpha=0.8, zorder=1)
        cfg_rotation_above = 0
        cfg_rotation_below = 0

        def measure_box_height(sample_text, rotation, bbox_style):
            probe = ax4.text(
                0, 0, sample_text, fontsize=FONTSIZE, rotation=rotation,
                va="center", ha="center", transform=ax4.transData, bbox=bbox_style,
            )
            fig4.canvas.draw()
            bbox = probe.get_window_extent(renderer=fig4.canvas.get_renderer()).transformed(ax4.transData.inverted())
            probe.remove()
            return abs(bbox.height)

        ts_box_height = measure_box_height("20000 s".center(min_width_ts + 2), 0, ts_box_style)
        cfg_box_height = measure_box_height("(7, 62.5)".center(min_width_cfg), 0, cfg_box_style)
        spacing = max(0.04 * y_range, 0.65 * ts_box_height)

        n_configs = len(dotted_positions)
        half = 15  # cfg/ts 0-14 above curve, 15-21 below
        dotted_with_labels = list(zip(dotted_positions, [format_sf_bw_label(l) for l in ["SF7_BW62.5"] + [lbl for _, lbl, _ in sf_bw_changes]]))

        ts1_even = ymax - spacing - ts_box_height
        ts1_odd = ts1_even - ts_box_height - spacing
        cfg1_7 = ts1_odd - spacing
        ts2_even = ts1_even
        ts2_odd = ts1_odd
        cfg8_15 = cfg1_7
        ts3_odd = ymin + spacing + ts_box_height
        ts3_even = ts3_odd + spacing + ts_box_height
        cfg_even_below = ts3_even + spacing
        cfg_odd_below = cfg_even_below + cfg_box_height + spacing

        cfg_even_off = cfg_box_height + spacing
        y_cfg = []
        y_ts = []
        for i in range(n_configs):
            if i < 7:
                y_ts.append(ts1_odd if i % 2 == 1 else ts1_even)
                y_cfg.append(cfg1_7 - cfg_even_off if i % 2 == 1 else cfg1_7)
            elif i < 8:
                y_ts.append(ts1_odd if i % 2 == 1 else ts1_even)
                y_cfg.append(cfg1_7 - cfg_even_off if i % 2 == 1 else cfg1_7)
            elif i < 15:
                y_ts.append(ts2_odd if i % 2 == 1 else ts2_even)
                y_cfg.append(cfg8_15 - cfg_even_off if i % 2 == 1 else cfg8_15)
            else:
                y_ts.append(ts3_odd if i % 2 == 1 else ts3_even)
                y_cfg.append(cfg_even_below if i % 2 == 1 else cfg_odd_below)

        max_ts_width = max(
            len(f"{np.interp(pkt_idx, indices_avg, times_smooth / 1000):.0f} s")
            for pkt_idx, _ in dotted_with_labels
        )
        ts_line_off = 0.18 * ts_box_height
        for i, (pkt_idx, label) in enumerate(dotted_with_labels):
            y_at_curve = np.interp(pkt_idx, indices_avg, times_smooth / 1000)
            label_padded = label.center(max(min_width_cfg, len(label)))
            ts_raw = f"{y_at_curve:.0f} s"
            ts_str = ts_raw.center(max_ts_width)
            if i < half:
                cfg_lo = y_cfg[i] - cfg_box_height * 0.92
                cfg_hi = y_cfg[i]
                y_line_end = y_ts[i] - ts_line_off
                ax4.plot([pkt_idx, pkt_idx], [y_at_curve, cfg_lo], **connector_style)
                ax4.plot([pkt_idx, pkt_idx], [cfg_hi, y_line_end], **connector_style)
                ax4.text(pkt_idx, y_cfg[i], label_padded, fontsize=FONTSIZE, rotation=cfg_rotation_above, va="top", ha="center", color="black",
                         bbox=cfg_box_style, zorder=15)
                ax4.text(pkt_idx, y_ts[i], ts_str, fontsize=FONTSIZE, va="bottom", ha="center", color="black",
                         bbox=ts_box_style, zorder=15)
            else:
                cfg_lo = y_cfg[i]
                cfg_hi = y_cfg[i] + cfg_box_height * 0.92
                y_line_start = y_ts[i] + ts_line_off
                ax4.plot([pkt_idx, pkt_idx], [y_line_start, cfg_lo], **connector_style)
                ax4.plot([pkt_idx, pkt_idx], [cfg_hi, y_at_curve], **connector_style)
                ax4.text(pkt_idx, y_cfg[i], label_padded, fontsize=FONTSIZE, rotation=cfg_rotation_below, va="bottom", ha="center", color="black",
                         bbox=cfg_box_style, zorder=15)
                ax4.text(pkt_idx, y_ts[i], ts_str, fontsize=FONTSIZE, va="top", ha="center", color="black",
                         bbox=ts_box_style, zorder=15)
        legend_elements = [
            Patch(facecolor=cfg_box_style["facecolor"], edgecolor=cfg_box_style["edgecolor"], label="(SF, BW)"),
            Patch(facecolor=ts_box_style["facecolor"], edgecolor=ts_box_style["edgecolor"], label=r"$T_{\mathrm{init}}$ at config"),
        ]
        ax4.legend(handles=legend_elements, loc="upper right", fontsize=FONTSIZE)
        fig4.tight_layout()
        fig4.savefig(os.path.join(out_dir, "raw_time_since_transmission_init_combined_v4.png"), dpi=220, bbox_inches="tight")
        plt.close()
        print("Saved: raw_time_since_transmission_init_combined_v4.png")
        # Save CSV to parent folder (raw_test_data_plots) to avoid breaking other scripts
        csv_dir = os.path.join(WORKSPACE, "results", "raw_test_data_plots")
        csv_path = os.path.join(csv_dir, "config_change_T_init.csv")
        config_rows = []
        for pkt_idx, sf, bw, tp, t_ms in all_config_entries:
            cfg = f"{sf}, {bw/1000}" if bw % 1000 else f"{sf}, {bw//1000}"
            config_rows.append({"packet_index": pkt_idx, "config": cfg, "TP": tp, "T_init_s": round(t_ms / 1000, 2)})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["packet_index", "config", "TP", "T_init_s"])
            w.writeheader()
            w.writerows(config_rows)
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
