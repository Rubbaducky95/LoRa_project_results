"""
Plot packet density heatmap: Energy vs Time for every packet.
Color = count of packets in each bin (density).
"""
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
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


def collect_all_packets_time_energy(data_root, filters=None):
    """
    Returns [(time_min, energy_mj), ...] for every packet.
    Uses internal timer (minutes) and energy per packet.
    """
    filters = filters or []
    pts = []
    dist_folders = sorted(
        [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith("distance_")],
        key=lambda x: parse_distance(x) or 0,
    )
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
            if any(config_matches_filter(sf, bw, tp, distance, f) for f in filters):
                continue
            path = find_file_in_dist(dpath, sf, bw, tp)
            if not path or not os.path.isfile(path):
                continue
            for raw_ms, lost, _, energy, tx_ms in file_packets(path, distance, sf, bw, tp):
                if prev_raw_ms is not None and raw_ms < prev_raw_ms:
                    timer_ms = 0
                time_min = timer_ms / 60000.0
                if energy is not None and energy > 0:
                    pts.append((time_min, energy))
                timer_ms += tx_ms
                prev_raw_ms = raw_ms

    return pts


def main():
    setup_plot_style()
    output_dir = os.path.join(WORKSPACE, "results", "raw_test_data_plots")
    os.makedirs(output_dir, exist_ok=True)
    filters = []
    n_time_bins = 120
    n_energy_bins = 100

    pts = collect_all_packets_time_energy(DATA_ROOT, filters)
    if not pts:
        print("No data found.")
        return

    all_times = [p[0] for p in pts]
    all_energies = [p[1] for p in pts]
    t_min, t_max = min(all_times), max(all_times)
    e_min, e_max = min(all_energies), max(all_energies)
    t_min = max(0, t_min - 0.5)
    e_min = max(0.01, e_min * 0.98)
    e_max = e_max * 1.02
    time_bins = np.linspace(t_min, t_max + 0.5, n_time_bins + 1)
    energy_bins = np.linspace(e_min, e_max, n_energy_bins + 1)

    # 2D histogram: count per bin
    mat, _, _ = np.histogram2d(
        [p[0] for p in pts],
        [p[1] for p in pts],
        bins=[time_bins, energy_bins],
    )
    mat = mat.T  # histogram2d returns (x, y) = (time, energy), we want energy rows x time cols

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(
        time_bins,
        energy_bins,
        mat,
        cmap="viridis",
        shading="flat",
    )
    ax.set_xlabel("Time (minutes, internal timer from first packet)")
    ax.set_ylabel("Energy per packet (mJ)")
    ax.set_title(f"Packet density: energy vs time ({len(pts):,} packets)")
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Packet count")
    out_path = os.path.join(output_dir, "raw_density_energy_vs_time.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: raw_density_energy_vs_time.png ({len(pts):,} packets)")


if __name__ == "__main__":
    main()
