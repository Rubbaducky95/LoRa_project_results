import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt


DATA_ROOT = r"C:\Users\ruben\Documents\LoRa Project\dataset"
OUTPUT_PNG = r"C:\Users\ruben\Documents\LoRa Project\results\dataset_plots\dataset_per_vs_bw.png"

BW_VALUES = [62500, 125000, 250000, 500000]
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
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


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
    if not rows:
        return None, None
    header = rows[0]
    if "payload" not in header:
        return None, None
    if "BW" not in header:
        return None, None
    payload_idx = header.index("payload")
    bw_idx = header.index("BW")

    total = 0
    lost = 0
    bw_value = None
    for r in rows[1:]:
        if len(r) <= max(payload_idx, bw_idx):
            continue
        if bw_value is None:
            try:
                bw_value = int(float(r[bw_idx]))
            except Exception:
                pass
        total += 1
        if not payload_is_valid(r[payload_idx]):
            lost += 1

    if total == 0 or bw_value is None:
        return None, None
    return bw_value, (lost / total) * 100.0


def collect_per_by_bw():
    grouped = defaultdict(list)
    for dn in sorted(os.listdir(DATA_ROOT)):
        dpath = os.path.join(DATA_ROOT, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        for root, _, files in os.walk(dpath):
            for fn in files:
                if not fn.endswith(".csv"):
                    continue
                bw, per = file_per(os.path.join(root, fn))
                if bw is None or per is None or bw not in BW_VALUES:
                    continue
                grouped[bw].append(per)
    return {bw: (sum(vals) / len(vals) if vals else None) for bw, vals in grouped.items()}


def main():
    setup_plot_style()
    os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
    summary = collect_per_by_bw()

    xs = []
    ys = []
    for bw in BW_VALUES:
        v = summary.get(bw)
        if v is None:
            continue
        xs.append(bw)
        ys.append(v)

    plt.figure(figsize=(6.2, 4.8))
    plt.bar(
        [str(x) for x in xs],
        ys,
        color=["#9467bd", "#1f77b4", "#2ca02c", "#d62728"][: len(xs)],
        edgecolor="black",
        linewidth=0.6,
        alpha=0.9,
    )
    plt.xlabel("Bandwidth (Hz)")
    plt.ylabel(r"Average PER (\%)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
