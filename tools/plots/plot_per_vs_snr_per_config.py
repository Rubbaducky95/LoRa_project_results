"""
Plot PER vs SNR for each configuration separately.
Requires snr_db column (run add_snr_to_raw_dataset.py first).
Outputs to raw_test_data_plots/per_vs_snr_all_configs/
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
            "lines.linewidth": 2,
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


def collect_per_snr_by_config(data_root):
    """Returns {(sf,bw,tp): [(snr, per), ...]} for each config."""
    by_cfg = defaultdict(list)
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
                path = os.path.join(root, fn)
                per = file_per(path)
                snr = file_avg_snr(path)
                if per is not None and snr is not None:
                    by_cfg[cfg].append((snr, per))
    return dict(by_cfg)


def main():
    parser = argparse.ArgumentParser(description="Plot PER vs SNR for each config.")
    parser.add_argument("--data-root", default=DATA_ROOT, help="Dataset root.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    args = parser.parse_args()
    if args.output_dir is None:
        base = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        args.output_dir = os.path.join(WORKSPACE, "results", base, "per_vs_snr_all_configs")

    setup_plot_style()
    os.makedirs(args.output_dir, exist_ok=True)
    by_cfg = collect_per_snr_by_config(args.data_root)

    for cfg, points in sorted(by_cfg.items()):
        if not points:
            continue
        sf, bw, tp = cfg
        snrs = np.array([p[0] for p in points])
        pers = np.array([p[1] for p in points]) * 100  # percent

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.scatter(snrs, pers, s=50, color="#1f77b4", alpha=0.8, edgecolors="black", linewidths=0.5)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("PER (%)")
        ax.set_title(f"SF{sf} BW{bw//1000} kHz TP{tp} dBm")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(6, pers.max() * 1.1))

        fig.tight_layout()
        fname = f"SF{sf}_BW{bw}_TP{tp}.png"
        out_path = os.path.join(args.output_dir, fname)
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()

    print(f"Saved {len(by_cfg)} plots to {args.output_dir}")


if __name__ == "__main__":
    main()
