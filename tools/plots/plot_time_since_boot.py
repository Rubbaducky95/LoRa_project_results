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
    """Convert 'SF7 BW62.5' or 'SF7_BW62.5' to '7, 62.5'."""
    s = raw_label.replace("_", " ")
    m = re.match(r"SF(\d+)\s*BW([\d.]+)", s)
    if m:
        return f"{m.group(1)}, {m.group(2)}"
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

    # # raw_time_since_transmission_init.png (commented out)
    # fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows), sharex=False, sharey=False, squeeze=False)
    # for ax in axes.flat:
    #     ax.set_visible(False)
    # for i, dn in enumerate(dist_folders):
    #     dpath = os.path.join(DATA_ROOT, dn)
    #     distance = parse_distance(dn)
    #     if distance is None:
    #         continue
    #     row, col = i // ncols, i % ncols
    #     ax = axes[row, col]
    #     ax.set_visible(True)
    #     indices, times, config_ids = collect_times_for_distance(dpath)
    #     if len(indices) == 0:
    #         continue
    #     sigma = min(5, max(1, len(times) // 100))
    #     times_smooth = gaussian_smooth(times, sigma=sigma)
    #     ax.plot(indices, times_smooth / 1000, "b-", linewidth=1.2, alpha=0.9)
    #     ax.set_title(dn, fontsize=9)
    #     ax.set_xlabel("Packet index")
    #     ax.set_ylabel("time_since_transmission_init (s)")
    #     ax.grid(True, alpha=0.3)
    # fig.suptitle("time_since_transmission_init_ms", fontsize=11)
    # fig.tight_layout()
    # fig.savefig(os.path.join(out_dir, "raw_time_since_transmission_init.png"), dpi=220, bbox_inches="tight")
    # plt.close()
    # print("Saved: raw_time_since_transmission_init.png")

    # # raw_time_since_boot.png (commented out)
    # fig2, axes2 = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows), sharex=False, sharey=False, squeeze=False)
    # for ax in axes2.flat:
    #     ax.set_visible(False)
    # for i, dn in enumerate(dist_folders):
    #     dpath = os.path.join(DATA_ROOT, dn)
    #     if parse_distance(dn) is None:
    #         continue
    #     row, col = i // ncols, i % ncols
    #     ax = axes2[row, col]
    #     ax.set_visible(True)
    #     indices, times, _ = collect_times_for_distance(dpath, time_col="time_since_boot_ms")
    #     if len(indices) == 0:
    #         continue
    #     sigma = min(5, max(1, len(times) // 100))
    #     times_smooth = gaussian_smooth(times, sigma=sigma)
    #     ax.plot(indices, times_smooth / 1000, "b-", linewidth=1.2, alpha=0.9)
    #     resets = reset_by_dist.get(dn, [])
    #     for pkt_idx, prev_ms, curr_ms in resets:
    #         if 0 <= pkt_idx < len(indices):
    #             ax.axvline(x=pkt_idx, color="r", linestyle="--", linewidth=0.8, alpha=0.6)
    #     if resets:
    #         ax.text(0.02, 0.98, f"{len(resets)} fixes", transform=ax.transAxes, fontsize=8, va="top",
    #                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    #     ax.set_title(dn, fontsize=9)
    #     ax.set_xlabel("Packet index")
    #     ax.set_ylabel("time_since_boot (s)")
    #     ax.grid(True, alpha=0.3)
    # fig2.suptitle("time_since_boot_ms: dotted lines = inconsistency locations", fontsize=11)
    # fig2.tight_layout()
    # fig2.savefig(os.path.join(out_dir, "raw_time_since_boot.png"), dpi=220, bbox_inches="tight")
    # plt.close()
    # print("Saved: raw_time_since_boot.png")

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

        xtick_positions = [0] + [p for p, _, _ in sf_bw_changes]
        xtick_labels = [l.replace("_", " ") for l in ["SF7_BW62.5"] + [lbl for _, lbl, _ in sf_bw_changes]]

        # # Combined plot v1 (commented out)
        # TEXTWIDTH_IN = 3.5
        # fig3, ax3 = plt.subplots(figsize=(2 * TEXTWIDTH_IN, 4))
        # ax3.plot(indices_avg, times_smooth / 1000, "b-", linewidth=1.5, alpha=0.9)
        # ax3.set_xticks(xtick_positions)
        # ax3.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=9)
        # ymin = ax3.get_ylim()[0]
        # dotted_positions = [0] + [p for p, _, _ in sf_bw_changes]
        # for pkt_idx in dotted_positions:
        #     y_at_curve = np.interp(pkt_idx, indices_avg, times_smooth / 1000)
        #     ax3.plot([pkt_idx, pkt_idx], [ymin, y_at_curve], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        # ax3.set_xlabel("Config (300 packets between changes)", fontsize=10)
        # ax3.set_ylabel("Time since first packet sent (s)", fontsize=10)
        # ax3.tick_params(axis="y", labelsize=9)
        # xmax = ax3.get_xlim()[1]
        # ax3.set_xlim(xmax=xmax - 300)
        # ax3.yaxis.set_major_formatter(plt.FuncFormatter(e3_format))
        # ax3.grid(True, alpha=0.3)
        # fig3.tight_layout()
        # fig3.savefig(os.path.join(out_dir, "raw_time_since_transmission_init_combined_v1.png"), dpi=220, bbox_inches="tight")
        # plt.close()
        # print("Saved: raw_time_since_transmission_init_combined_v1.png")

        # Combined plot v3: packet index on X-axis, dotted lines from curve up, labels black and rotated
        FONTSIZE = IEEE_FONTSIZE
        dotted_positions = [0] + [p for p, _, _ in sf_bw_changes]
        fig4, ax4 = plt.subplots(figsize=FIGSIZE_TWO_COL)
        ax4.plot(indices_avg, times_smooth / 1000, "b-", linewidth=2.2, alpha=0.9)
        ax4.set_xlabel("Packet index", fontsize=FONTSIZE)
        ax4.set_ylabel(r"$T_{\mathrm{init}}$ = Time since first packet sent (s)", fontsize=FONTSIZE)
        ax4.tick_params(axis="both", labelsize=FONTSIZE)
        xmax = ax4.get_xlim()[1]
        ax4.set_xlim(xmax=xmax - 300)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(e3_format))
        ax4.grid(True, alpha=0.3)
        ymax = ax4.get_ylim()[1]
        ymin = ax4.get_ylim()[0]
        line_height_short = 0.14 * (ymax - ymin)  # increased: more room above
        line_height_tall = 0.32 * (ymax - ymin)
        line_height_short_below = 0.12 * (ymax - ymin)
        line_height_tall_below = 0.30 * (ymax - ymin)
        timestamp_extension = 0.06 * (ymax - ymin)  # extra segment for timestamp above
        y_range = ymax - ymin
        ts_line_end_offset_above = 0.015 * y_range  # line ends a bit below timestamp (pos unchanged)
        y_ts_odd_above = ymax - 0.10 * y_range
        y_ts_even_above = ymax - 0.18 * y_range
        y_ts_odd_below = ymin + 0.15 * y_range
        y_ts_even_below = ymin + 0.07 * y_range
        ts_line_start_offset_below = 0.015 * y_range  # line starts a bit above timestamp (pos unchanged)
        y_cfg_odd_above = ymax - 0.28 * y_range
        y_cfg_even_above = ymax - 0.46 * y_range
        y_cfg_odd_below = ymin + 0.43 * y_range
        y_cfg_even_below = ymin + 0.25 * y_range
        # Measure config text box height (representative label)
        _t = ax4.text(0, 0, "7, 62.5", fontsize=FONTSIZE, rotation=45, va="center", ha="center", transform=ax4.transData)
        fig4.canvas.draw()
        _bb = _t.get_window_extent(renderer=fig4.canvas.get_renderer()).transformed(ax4.transData.inverted())
        config_text_height = abs(_bb.height)
        _t.remove()
        n_configs = len(dotted_positions)
        half = n_configs // 2 + 4  # move one more from below to above
        dotted_with_labels = list(zip(dotted_positions, [format_sf_bw_label(l) for l in ["SF7_BW62.5"] + [lbl for _, lbl, _ in sf_bw_changes]]))
        for i, (pkt_idx, label) in enumerate(dotted_with_labels):
            y_at_curve = np.interp(pkt_idx, indices_avg, times_smooth / 1000)
            if y_at_curve >= 10000:
                ts_str = f"{y_at_curve/1000:.1f}e3"
            elif y_at_curve < 100:
                ts_str = f"   {y_at_curve:.0f}   "
            elif y_at_curve < 1000:
                ts_str = f"  {y_at_curve:.0f}  "
            else:
                ts_str = f" {y_at_curve:.0f} "
            if i < half:
                line_height_base = line_height_tall if i % 2 == 1 else line_height_short
                ts_ext = -config_text_height * 0.3 if i == half - 1 else timestamp_extension  # last of above: much shorter timestamp line
                total_line = line_height_base + config_text_height + ts_ext
            else:
                # Below: start with short; last 3 use room below (taller)
                below_idx = i - half
                is_last_three = i >= n_configs - 3
                if is_last_three:
                    line_height_base = 0.38 * (ymax - ymin)  # use room below
                else:
                    line_height_base = (line_height_short_below if below_idx % 2 == 0 else line_height_tall_below)
                total_line = line_height_base + config_text_height
            if i < half:
                y_label = y_cfg_odd_above if i % 2 == 1 else y_cfg_even_above
                y_end = y_ts_odd_above if i % 2 == 1 else y_ts_even_above
                cfg_lo = y_label - config_text_height * 0.4
                cfg_hi = y_label + config_text_height * 0.4
                y_line_end = y_end - ts_line_end_offset_above  # line stops below timestamp
                ax4.plot([pkt_idx, pkt_idx], [y_at_curve, cfg_lo], color="gray", linestyle="--", linewidth=0.8, alpha=0.7, zorder=10)
                ax4.plot([pkt_idx, pkt_idx], [cfg_hi, y_line_end], color="gray", linestyle="--", linewidth=0.8, alpha=0.7, zorder=10)
                ax4.text(pkt_idx, y_label, label, fontsize=FONTSIZE, rotation=45, va="center", ha="center", color="black",
                         bbox=dict(boxstyle="round,pad=0.1", facecolor="lightgray", edgecolor="black", alpha=0.9))
                ax4.text(pkt_idx, y_end, ts_str, fontsize=FONTSIZE, va="bottom", ha="center", color="black",
                         bbox=dict(boxstyle="round,pad=0.1", facecolor="yellow", edgecolor="black", alpha=0.9))
            else:
                y_label = y_cfg_odd_below if i % 2 == 1 else y_cfg_even_below
                y_end = y_ts_odd_below if i % 2 == 1 else y_ts_even_below
                cfg_lo = y_label - config_text_height * 0.4
                cfg_hi = y_label + config_text_height * 0.4
                y_line_start = y_end + ts_line_start_offset_below
                ax4.plot([pkt_idx, pkt_idx], [y_line_start, cfg_lo], color="gray", linestyle="--", linewidth=0.8, alpha=0.7, zorder=10)
                ax4.plot([pkt_idx, pkt_idx], [cfg_hi, y_at_curve], color="gray", linestyle="--", linewidth=0.8, alpha=0.7, zorder=10)
                ax4.text(pkt_idx, y_label, label, fontsize=FONTSIZE, rotation=45, va="center", ha="center", color="black",
                         bbox=dict(boxstyle="round,pad=0.1", facecolor="lightgray", edgecolor="black", alpha=0.9))
                ax4.text(pkt_idx, y_end, ts_str, fontsize=FONTSIZE, va="top", ha="center", color="black",
                         bbox=dict(boxstyle="round,pad=0.1", facecolor="yellow", edgecolor="black", alpha=0.9))
        legend_elements = [
            Patch(facecolor="lightgray", edgecolor="black", label="SF, BW"),
            Patch(facecolor="yellow", edgecolor="black", label=r"$T_{\mathrm{init}}$ at config"),
        ]
        ax4.legend(handles=legend_elements, loc="upper right", fontsize=FONTSIZE)
        fig4.tight_layout()
        fig4.savefig(os.path.join(out_dir, "raw_time_since_transmission_init_combined_v3.png"), dpi=220, bbox_inches="tight")
        plt.close()
        print("Saved: raw_time_since_transmission_init_combined_v3.png")
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
