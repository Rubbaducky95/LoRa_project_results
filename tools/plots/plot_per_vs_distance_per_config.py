"""
Plot average PER vs distance for each configuration separately.
Outputs to raw_test_data_plots/per_all_configs/ (or dataset_plots/per_all_configs/).
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
XTICK_DISTANCES = [6.25, 25, 50, 75, 100]


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


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def collect_per_by_config(data_root):
    """Returns {(sf,bw,tp): [(distance, per), ...]} for each config."""
    by_cfg = defaultdict(list)
    for dn in sorted(os.listdir(data_root)):
        dpath = os.path.join(data_root, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        distance = parse_distance(dn)
        if distance is None:
            continue
        for root, _, files in os.walk(dpath):
            for fn in files:
                cfg = parse_cfg(fn)
                if cfg is None:
                    continue
                sf, bw, tp = cfg
                if bw not in BW_VALUES or tp not in TP_VALUES or sf not in SF_VALUES:
                    continue
                per = file_per(os.path.join(root, fn))
                if per is not None:
                    by_cfg[cfg].append((distance, per))
    for cfg in by_cfg:
        by_cfg[cfg].sort(key=lambda x: x[0])
    return dict(by_cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Plot PER vs distance for each config separately."
    )
    parser.add_argument("--data-root", default=DATA_ROOT, help="Dataset root.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    args = parser.parse_args()
    if args.output_dir is None:
        base = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        args.output_dir = os.path.join(WORKSPACE, "results", base, "per_all_configs")

    setup_plot_style()
    os.makedirs(args.output_dir, exist_ok=True)
    by_cfg = collect_per_by_config(args.data_root)

    for cfg, points in sorted(by_cfg.items()):
        if not points:
            continue
        sf, bw, tp = cfg
        distances = np.array([p[0] for p in points])
        pers = np.array([p[1] for p in points]) * 100  # percent

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.plot(distances, pers, "o-", color="#1f77b4", linewidth=2, markersize=6)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Average PER (%)")
        ax.set_title(f"SF{sf} BW{bw//1000} kHz TP{tp} dBm")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(6, pers.max() * 1.1))
        ticks = [t for t in XTICK_DISTANCES if min(distances) - 1e-6 <= t <= max(distances) + 1e-6]
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:.2f}".rstrip("0").rstrip(".") for t in ticks], rotation=45)

        fig.tight_layout()
        fname = f"SF{sf}_BW{bw}_TP{tp}.png"
        out_path = os.path.join(args.output_dir, fname)
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()

    print(f"Saved {len(by_cfg)} plots to {args.output_dir}")


if __name__ == "__main__":
    main()
