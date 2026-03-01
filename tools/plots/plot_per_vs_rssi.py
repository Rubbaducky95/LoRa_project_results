"""
Plot PER vs average RSSI. Each point is one config at one distance.
"""
import argparse
import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
DATA_ROOT = os.path.join(WORKSPACE, "raw_test_data")

BW_VALUES = [62500, 125000, 250000, 500000]
TP_VALUES = [2, 12, 22]
SF_VALUES = [7, 8, 9, 10, 11, 12]
HEX_RE = re.compile(r"^[0-9A-F]+$")
OUTLIER_CONFIG = (10, 62500, 22)


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
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    return tuple(map(int, m.groups())) if m else None


def payload_is_valid(payload):
    if payload == "PACKET_LOST":
        return False
    parts = (payload or "").strip('"').split(",")
    for part in parts:
        if part and HEX_RE.match(part) is None:
            return False
    return True


def file_per_and_rssi(path):
    """Returns (per, avg_rssi) or None. Uses RSSI from all data rows (received + lost)."""
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "payload" not in rows[0]:
        return None
    header = rows[0]
    payload_idx = header.index("payload")
    rssi_idx = header.index("rssi_corrected") if "rssi_corrected" in header else (header.index("rssi") if "rssi" in header else None)
    if rssi_idx is None:
        return None
    total = 0
    lost = 0
    rssi_vals = []
    for r in rows[1:]:
        if len(r) <= max(payload_idx, rssi_idx):
            continue
        if str(r[payload_idx]).strip().startswith("CFG "):
            continue
        total += 1
        if not payload_is_valid(r[payload_idx]):
            lost += 1
        try:
            rssi_vals.append(float(r[rssi_idx]))
        except (ValueError, TypeError):
            pass
    if total == 0 or not rssi_vals:
        return None
    per = lost / total
    avg_rssi = np.mean(rssi_vals)
    return per, avg_rssi


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def collect_data(data_root):
    """Returns list of (rssi, per) points, excluding outlier config."""
    points = []
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
                if bw not in BW_VALUES or tp not in TP_VALUES or sf not in SF_VALUES:
                    continue
                if cfg == OUTLIER_CONFIG:
                    continue
                result = file_per_and_rssi(os.path.join(root, fn))
                if result is not None:
                    per, avg_rssi = result
                    points.append((avg_rssi, per))
    return points


def collect_data_by_bw(data_root):
    """Returns {bw: [(rssi, per), ...]} excluding outlier."""
    by_bw = defaultdict(list)
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
                if bw not in BW_VALUES or tp not in TP_VALUES or sf not in SF_VALUES:
                    continue
                if cfg == OUTLIER_CONFIG:
                    continue
                result = file_per_and_rssi(os.path.join(root, fn))
                if result is not None:
                    per, avg_rssi = result
                    by_bw[bw].append((avg_rssi, per))
    return dict(by_bw)


def main():
    parser = argparse.ArgumentParser(description="Plot PER vs average RSSI.")
    parser.add_argument("--data-root", default=DATA_ROOT, help="Dataset root.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--binned", action="store_true", help="Show binned average instead of scatter.")
    parser.add_argument("--by-bw", action="store_true", help="4 quadrants, one per BW (x=RSSI, y=PER).")
    parser.add_argument("--circle", action="store_true", help="Single circle: 4 quadrants by BW, radius=PER, angle=RSSI.")
    args = parser.parse_args()
    if args.output_dir is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        args.output_dir = os.path.join(WORKSPACE, "results", out_dir)

    setup_plot_style()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.circle:
        by_bw = collect_data_by_bw(args.data_root)
        if not by_bw:
            print("No data found.")
            return
        all_rssi = []
        for pts in by_bw.values():
            all_rssi.extend([p[0] for p in pts])
        rssi_min, rssi_max = min(all_rssi), max(all_rssi)
        if rssi_max <= rssi_min:
            rssi_max = rssi_min + 1
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
        colors = {62500: "#9467bd", 125000: "#1f77b4", 250000: "#2ca02c", 500000: "#d62728"}
        base_angles = {62500: 0, 125000: np.pi / 2, 250000: np.pi, 500000: 3 * np.pi / 2}
        for bw in BW_VALUES:
            pts = by_bw.get(bw, [])
            if not pts:
                continue
            thetas, rs = [], []
            for rssi, per in pts:
                frac = (rssi - rssi_min) / (rssi_max - rssi_min)
                theta = base_angles[bw] + frac * (np.pi / 2)
                r = per * 6  # 0-6% scale (100% PER -> 6)
                thetas.append(theta)
                rs.append(r)
            ax.scatter(thetas, rs, s=50, c=colors[bw], alpha=0.8, edgecolors="black", linewidths=0.5, label=f"BW {bw//1000} kHz")
        for r in [2, 4, 6]:
            ax.plot(np.linspace(0, 2 * np.pi, 100), [r] * 100, "k-", alpha=0.2, linewidth=0.6)
        for t in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
            ax.plot([t] * 2, [0, 6], "k-", alpha=0.2, linewidth=0.6)
        ax.set_ylim(0, 6.5)
        ax.set_ylabel("PER (%)", labelpad=30)
        ax.set_yticks([0, 2, 4, 6])
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        fig.tight_layout()
        prefix = "dataset" if "dataset" in args.data_root else "raw"
        fname = f"{prefix}_per_vs_rssi_circle.png"
        out_path = os.path.join(args.output_dir, fname)
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")
        return

    if args.by_bw:
        by_bw = collect_data_by_bw(args.data_root)
        if not by_bw:
            print("No data found.")
            return
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
        axes = axes.flatten()
        colors = {62500: "#9467bd", 125000: "#1f77b4", 250000: "#2ca02c", 500000: "#d62728"}
        for i, bw in enumerate(BW_VALUES):
            ax = axes[i]
            pts = by_bw.get(bw, [])
            if pts:
                rssis = np.array([p[0] for p in pts])
                pers = np.array([p[1] for p in pts]) * 100
                ax.scatter(rssis, pers, s=50, c=colors[bw], alpha=0.8, edgecolors="black", linewidths=0.5)
            ax.set_xlabel("Average RSSI (dBm)")
            ax.set_ylabel("PER (%)")
            ax.set_title(f"BW {bw // 1000} kHz", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()
        fig.tight_layout()
        prefix = "dataset" if "dataset" in args.data_root else "raw"
        fname = f"{prefix}_per_vs_rssi_by_bw.png"
        out_path = os.path.join(args.output_dir, fname)
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")
        return

    points = collect_data(args.data_root)
    if not points:
        print("No data found.")
        return

    rssis = np.array([p[0] for p in points])
    pers = np.array([p[1] for p in points]) * 100  # percent

    fig, ax = plt.subplots(figsize=(6.5, 4.2))

    if args.binned:
        # Bin by RSSI, compute mean PER per bin
        n_bins = 15
        bins = np.linspace(rssis.min(), rssis.max(), n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_per = []
        bin_rssi = []
        for i in range(n_bins):
            mask = (rssis >= bins[i]) & (rssis < bins[i + 1])
            if mask.sum() > 0:
                bin_per.append(np.mean(pers[mask]))
                bin_rssi.append(bin_centers[i])
        ax.plot(bin_rssi, bin_per, "o-", color="#1f77b4", linewidth=2, markersize=6)
    else:
        ax.scatter(rssis, pers, alpha=0.6, s=40, color="#1f77b4", edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Average RSSI (dBm)")
    ax.set_ylabel("PER (%)")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Stronger signal (less negative) on the right

    fig.tight_layout()
    prefix = "dataset" if "dataset" in args.data_root else "raw"
    fname = f"{prefix}_per_vs_rssi.png"
    if args.binned:
        fname = fname.replace(".png", "_binned.png")
    out_path = os.path.join(args.output_dir, fname)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


if __name__ == "__main__":
    main()
