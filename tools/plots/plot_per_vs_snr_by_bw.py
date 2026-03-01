"""
Plot PER vs SNR: 4 subplots (one per BW), each with lines for all (SF, TP) configs.
Each line = trajectory through (SNR, PER) at each distance point.
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
            "lines.linewidth": 2,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    return tuple(map(int, m.groups())) if m else None


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


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


def collect_data(data_root):
    """Returns {(sf,bw,tp): [(distance, snr, per), ...]} and {(distance,bw): [(sf,tp,snr,per),...]}."""
    by_cfg = defaultdict(list)
    by_dist_bw = defaultdict(list)
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
                if cfg == OUTLIER_CONFIG:
                    continue
                path = os.path.join(root, fn)
                per = file_per(path)
                snr = file_avg_snr(path)
                if per is not None and snr is not None:
                    by_cfg[cfg].append((distance, snr, per))
                    by_dist_bw[(distance, bw)].append((sf, tp, snr, per))
    for cfg in by_cfg:
        by_cfg[cfg].sort(key=lambda x: x[0])
    return dict(by_cfg), dict(by_dist_bw)


def main():
    parser = argparse.ArgumentParser(description="Plot PER vs SNR by BW, lines per (SF,TP).")
    parser.add_argument("--data-root", default=DATA_ROOT, help="Dataset root.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--per-distance", action="store_true", help="One figure per distance (scatter).")
    args = parser.parse_args()
    if args.output_dir is None:
        base = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        args.output_dir = os.path.join(WORKSPACE, "results", base)

    setup_plot_style()
    os.makedirs(args.output_dir, exist_ok=True)
    by_cfg, by_dist_bw = collect_data(args.data_root)

    if args.per_distance:
        out_sub = os.path.join(args.output_dir, "per_vs_snr_by_bw_per_distance")
        os.makedirs(out_sub, exist_ok=True)
        distances = sorted(set(d for d, _ in by_dist_bw.keys()))
        colors_sf = {sf: plt.cm.tab10(i % 10) for i, sf in enumerate(SF_VALUES)}
        markers_tp = {2: "s", 12: "o", 22: "^"}
        for distance in distances:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
            axes = axes.flatten()
            for i, bw in enumerate(BW_VALUES):
                ax = axes[i]
                pts = by_dist_bw.get((distance, bw), [])
                for sf, tp, snr, per in pts:
                    ax.scatter(snr, per * 100, s=60, c=[colors_sf[sf]], marker=markers_tp[tp],
                              alpha=0.9, edgecolors="black", linewidths=0.5, label=f"SF{sf} TP{tp}")
                ax.set_xlabel("SNR (dB)")
                ax.set_ylabel("PER (%)")
                ax.set_title(f"BW {bw // 1000} kHz")
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 6)
                ax.legend(loc="upper right", fontsize=6, ncol=2)
            fig.suptitle(f"PER vs SNR at {distance}m", fontsize=12, y=1.02)
            fig.tight_layout()
            prefix = "dataset" if "dataset" in args.data_root else "raw"
            d_str = str(distance).replace(".", "p")
            out_path = os.path.join(out_sub, f"{prefix}_per_vs_snr_{d_str}m.png")
            fig.savefig(out_path, dpi=220, bbox_inches="tight")
            plt.close()
        print(f"Saved {len(distances)} figures to {out_sub}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    colors_sf = {sf: plt.cm.tab10(i % 10) for i, sf in enumerate(SF_VALUES)}
    tp_styles = {2: {"linestyle": "--", "alpha": 0.7, "linewidth": 1.5}, 12: {"linestyle": "-", "alpha": 1.0, "linewidth": 2}, 22: {"linestyle": ":", "alpha": 0.7, "linewidth": 1.5}}

    for i, bw in enumerate(BW_VALUES):
        ax = axes[i]
        for sf in SF_VALUES:
            for tp in TP_VALUES:
                cfg = (sf, bw, tp)
                pts = by_cfg.get(cfg, [])
                if not pts:
                    continue
                snrs = [p[1] for p in pts]
                pers = [p[2] * 100 for p in pts]
                ax.plot(snrs, pers, marker="o", color=colors_sf[sf], label=f"SF{sf} TP{tp}", **tp_styles[tp])
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("PER (%)")
        ax.set_title(f"BW {bw // 1000} kHz")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 6)
        ax.legend(loc="upper right", fontsize=6, ncol=2)

    fig.tight_layout()
    prefix = "dataset" if "dataset" in args.data_root else "raw"
    out_path = os.path.join(args.output_dir, f"{prefix}_per_vs_snr_by_bw.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {prefix}_per_vs_snr_by_bw.png")


if __name__ == "__main__":
    main()
