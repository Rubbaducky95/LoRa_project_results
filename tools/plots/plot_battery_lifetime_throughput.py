"""
Plot battery lifetime (assuming fixed capacity) and throughput vs average power.
Requires: energy per packet, packet count, time, payload size from raw CSVs.
"""
import argparse
import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
from plot_config import FIGSIZE_ONE_COL, FIGSIZE_TWO_COL, IEEE_FONTSIZE, SAVE_DPI

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


def read_energy_time_payload(path):
    """
    Return (sum_mj, count, duration_s, total_bits) or None.
    duration_s = (last - first) time_since_transmission_init_ms / 1000.
    """
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows:
        return None
    header = rows[0]
    if "energy_per_packet_min_mj" in header and "energy_per_packet_max_mj" in header:
        i_min = header.index("energy_per_packet_min_mj")
        i_max = header.index("energy_per_packet_max_mj")
        scale = 1.0
    elif "energy_per_packet_j_min" in header and "energy_per_packet_j_max" in header:
        i_min = header.index("energy_per_packet_j_min")
        i_max = header.index("energy_per_packet_j_max")
        scale = 1000.0
    else:
        return None
    i_time = header.index("time_since_transmission_init_ms") if "time_since_transmission_init_ms" in header else None
    i_payload = header.index("payload_size_bytes") if "payload_size_bytes" in header else None

    mids_raw = []
    times_ms = []
    payload_bytes = []
    for row in rows[1:]:
        if len(row) <= max(i_min, i_max):
            continue
        e_min = parse_float(row[i_min])
        e_max = parse_float(row[i_max])
        if e_min is None or e_max is None:
            continue
        mids_raw.append((e_min + e_max) / 2.0)
        if i_time is not None and len(row) > i_time:
            t = parse_float(row[i_time])
            if t is not None:
                times_ms.append(t)
        if i_payload is not None and len(row) > i_payload:
            p = parse_float(row[i_payload])
            if p is not None and p > 0:
                payload_bytes.append(int(p))

    if not mids_raw:
        return None
    if scale == 1.0:
        sample = mids_raw[: min(20, len(mids_raw))]
        sample_avg = sum(sample) / len(sample)
        if sample_avg < 1.0:
            scale = 1000.0
    mids_mj = [m * scale for m in mids_raw]
    sum_mj = sum(mids_mj)
    count = len(mids_mj)

    duration_s = 0.0
    if len(times_ms) >= 2:
        duration_s = (max(times_ms) - min(times_ms)) / 1000.0
    elif len(times_ms) == 1:
        duration_s = 0.001

    payload_b = int(sum(payload_bytes) / len(payload_bytes)) if payload_bytes else 37
    total_bits = count * payload_b * 8

    return sum_mj, count, duration_s, total_bits


def collect_power_throughput_data(data_root):
    """
    Per (tp, bw, sf): (sum_mj, total_duration_s, total_bits).
    Aggregated over all distance folders.
    """
    acc = defaultdict(lambda: (0.0, 0.0, 0))
    for dn in sorted(os.listdir(data_root)):
        dpath = os.path.join(data_root, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        for root, _, files in os.walk(dpath):
            for fn in files:
                cfg = parse_cfg(fn)
                if cfg is None:
                    continue
                sf, bw, tp = cfg
                if sf not in SF_VALUES or bw not in BW_VALUES or tp not in TP_VALUES:
                    continue
                result = read_energy_time_payload(os.path.join(root, fn))
                if result is None:
                    continue
                sum_mj, count, duration_s, total_bits = result
                if duration_s <= 0:
                    continue
                prev_mj, prev_dur, prev_bits = acc[(tp, bw, sf)]
                acc[(tp, bw, sf)] = (prev_mj + sum_mj, prev_dur + duration_s, prev_bits + total_bits)
    return dict(acc)


def compute_battery_lifetime(sum_mj, duration_s, capacity_wh):
    """Battery lifetime in days. power_W = sum_mj/1000 / duration_s."""
    if sum_mj <= 0 or duration_s <= 0:
        return 0.0
    power_W = (sum_mj / 1000.0) / duration_s
    lifetime_h = capacity_wh / power_W
    return lifetime_h / 24.0


def compute_avg_power_mw(sum_mj, duration_s):
    """Average power in mW."""
    if duration_s <= 0:
        return 0.0
    return (sum_mj / 1000.0) / duration_s * 1000.0


def compute_throughput_bps(total_bits, duration_s):
    """Throughput in bits/s."""
    if duration_s <= 0:
        return 0.0
    return total_bits / duration_s


def compute_energy_per_bit_uj(sum_mj, total_bits):
    """Energy per bit in µJ/bit."""
    if total_bits <= 0:
        return 0.0
    return sum_mj * 1000.0 / total_bits


def plot_battery_lifetime(data, output_png, capacity_wh=240.0):
    """Grouped bar: battery lifetime (days) per config, one bar per SF."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    lifetime_map = {}
    for (tp, bw, sf), (sum_mj, duration_s, total_bits) in data.items():
        lifetime_map[(tp, bw, sf)] = compute_battery_lifetime(sum_mj, duration_s, capacity_wh)

    configs = []
    for bw in BW_VALUES:
        for tp in TP_VALUES:
            sf_vals = [lifetime_map.get((tp, bw, sf), 0) for sf in SF_VALUES]
            max_lifetime = max(sf_vals) if sf_vals else 0
            label = f"{bw/1000:.1f}".rstrip("0").rstrip(".") + "-TP" + str(tp)
            configs.append((max_lifetime, label, sf_vals))
    configs.sort(key=lambda c: c[0])
    labels = [c[1] for c in configs]
    n_configs = len(configs)
    n_sf = len(SF_VALUES)
    bar_width = 0.12
    gap = 0.02
    group_width = n_sf * bar_width + (n_sf - 1) * gap
    x = np.arange(n_configs)
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_TWO_COL)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_sf))
    for i, sf in enumerate(SF_VALUES):
        offsets = (i - (n_sf - 1) / 2) * (bar_width + gap)
        vals = [configs[j][2][i] for j in range(n_configs)]
        ax.bar(x + offsets, vals, bar_width, label=f"SF{sf}", color=colors[i])
    ax.set_xlabel("Config (BW-TP)", fontsize=IEEE_FONTSIZE)
    ax.set_ylabel(f"Battery lifetime (days, {capacity_wh:.0f} Wh)", fontsize=IEEE_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlim(-0.6, n_configs - 0.4)
    ax.legend(loc="upper right", fontsize=IEEE_FONTSIZE, ncol=3)
    ax.set_ylim(bottom=0)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.22)
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_png}")


def plot_throughput_vs_power(data, output_png):
    """Scatter: throughput (bits/s) vs energy per bit (µJ/bit). Configs = SF-BW, TP values averaged."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    # Aggregate by (sf, bw): sum over TP, then compute energy/bit and throughput
    agg = defaultdict(lambda: (0.0, 0.0, 0))
    for (tp, bw, sf), (sum_mj, duration_s, total_bits) in data.items():
        if duration_s <= 0 or total_bits <= 0:
            continue
        prev_mj, prev_dur, prev_bits = agg[(sf, bw)]
        agg[(sf, bw)] = (prev_mj + sum_mj, prev_dur + duration_s, prev_bits + total_bits)

    points = []
    for (sf, bw), (sum_mj, duration_s, total_bits) in agg.items():
        if total_bits <= 0 or duration_s <= 0:
            continue
        energy_per_bit_uj = compute_energy_per_bit_uj(sum_mj, total_bits)
        throughput_bps = compute_throughput_bps(total_bits, duration_s)
        points.append((sf, bw, energy_per_bit_uj, throughput_bps))

    if not points:
        raise RuntimeError("No usable data for throughput vs power.")

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_ONE_COL)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(SF_VALUES)))
    for sf_idx, sf in enumerate(SF_VALUES):
        pts = [p for p in points if p[0] == sf]
        if not pts:
            continue
        x_vals = [p[2] for p in pts]
        y_vals = [p[3] for p in pts]
        ax.scatter(x_vals, y_vals, c=[colors[sf_idx]], label=f"SF{sf}", s=40, marker="o", edgecolors="k", linewidths=0.5)
    ax.set_xlabel(r"Energy per bit ($\mu$J/bit)", fontsize=IEEE_FONTSIZE)
    ax.set_ylabel("Throughput (bits/s)", fontsize=IEEE_FONTSIZE)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=IEEE_FONTSIZE, ncol=1, frameon=True)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    fig.subplots_adjust(left=0.14, right=0.72, top=0.95, bottom=0.12)
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot battery lifetime and throughput vs average power."
    )
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
    parser.add_argument(
        "--battery-wh",
        type=float,
        default=240.0,
        help="Battery capacity in Wh (default: 240).",
    )
    parser.add_argument(
        "--plot",
        choices=["both", "battery", "throughput"],
        default="both",
        help="Which plot(s) to generate.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        args.output_dir = os.path.join(WORKSPACE, "results", out_dir, "energy_bw_vs_sf_with_tp")

    setup_plot_style()
    data = collect_power_throughput_data(args.data_root)

    if args.plot in ("both", "battery"):
        plot_battery_lifetime(
            data,
            os.path.join(args.output_dir, f"raw_battery_lifetime_{args.battery_wh:.0f}wh.png"),
            capacity_wh=args.battery_wh,
        )
    if args.plot in ("both", "throughput"):
        plot_throughput_vs_power(
            data,
            os.path.join(args.output_dir, "raw_throughput_vs_energy_per_bit.png"),
        )


if __name__ == "__main__":
    main()
