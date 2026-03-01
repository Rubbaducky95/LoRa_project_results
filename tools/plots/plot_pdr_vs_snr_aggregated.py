"""
Plot PDR vs SNR with all BWs and distances clumped together.
PDR = 1 - PER (Packet Delivery Rate).
Fit a sigmoid/logistic curve to find the relationship.
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


def file_per(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "payload" not in rows[0]:
        return None
    header = rows[0]
    payload_idx = header.index("payload")
    total = 0
    lost = 0
    for r in rows[1:]:
        if len(r) <= payload_idx:
            continue
        if str(r[payload_idx]).strip().startswith("CFG "):
            continue
        total += 1
        if not payload_is_valid(r[payload_idx]):
            lost += 1
    if total == 0:
        return None
    return lost / total  # fraction 0-1


def file_avg_snr(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "snr_db" not in rows[0]:
        return None
    header = rows[0]
    snr_idx = header.index("snr_db")
    payload_idx = header.index("payload") if "payload" in header else -1
    vals = []
    for r in rows[1:]:
        if len(r) <= snr_idx:
            continue
        if payload_idx >= 0 and len(r) > payload_idx and str(r[payload_idx]).strip().startswith("CFG "):
            continue
        try:
            v = float(r[snr_idx])
            vals.append(v)
        except (ValueError, TypeError):
            pass
    if not vals:
        return None
    return np.mean(vals)




def collect_all_points(data_root):
    """Returns [(snr, pdr), ...] for all configs, all distances, clumped."""
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
                path = os.path.join(root, fn)
                per = file_per(path)
                snr = file_avg_snr(path)
                if per is not None and snr is not None:
                    pdr = 1.0 - per
                    points.append((snr, pdr))
    return points


def main():
    parser = argparse.ArgumentParser(description="Plot PDR vs SNR (all BWs/distances clumped).")
    parser.add_argument("--data-root", default=DATA_ROOT, help="Dataset root.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    args = parser.parse_args()
    if args.output_dir is None:
        base = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        args.output_dir = os.path.join(WORKSPACE, "results", base)

    setup_plot_style()
    os.makedirs(args.output_dir, exist_ok=True)
    points = collect_all_points(args.data_root)

    if not points:
        print("No data found. Run add_snr_to_raw_dataset.py first.")
        return

    snrs = np.array([p[0] for p in points])
    pdrs = np.array([p[1] for p in points]) * 100  # percent

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(snrs, pdrs, alpha=0.5, s=50, color="#1f77b4", edgecolors="black", linewidths=0.5)

    # Bin by SNR and plot mean PDR curve
    n_bins = 25
    bins = np.linspace(snrs.min(), snrs.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_stds = []
    for i in range(n_bins):
        mask = (snrs >= bins[i]) & (snrs < bins[i + 1])
        if mask.sum() > 0:
            bin_means.append(np.mean(pdrs[mask]))
            bin_stds.append(np.std(pdrs[mask]) if mask.sum() > 1 else 0)
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    bin_centers = np.array(bin_centers)
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    valid = ~np.isnan(bin_means)
    ax.plot(bin_centers[valid], bin_means[valid], "r-", linewidth=2.5, label="Mean (binned)")
    ax.fill_between(
        bin_centers[valid],
        bin_means[valid] - bin_stds[valid],
        np.minimum(100, bin_means[valid] + bin_stds[valid]),
        color="red",
        alpha=0.2,
    )
    ax.legend(loc="lower right")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("PDR (%)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    prefix = "dataset" if "dataset" in args.data_root else "raw"
    out_path = os.path.join(args.output_dir, f"{prefix}_pdr_vs_snr_aggregated.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {prefix}_pdr_vs_snr_aggregated.png")


if __name__ == "__main__":
    main()
