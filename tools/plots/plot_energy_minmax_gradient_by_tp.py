import argparse
import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
import numpy as np
from plot_config import FIGSIZE_TWO_COL, IEEE_FONTSIZE, SAVE_DPI
from matplotlib.colors import LinearSegmentedColormap

# Greyscale with slight offset on low end: low values = dark grey, high values = black
GREY_OFFSET_CMAP = LinearSegmentedColormap.from_list(
    "grey_offset", [(0.9, 0.9, 0.9), (0, 0, 0)]
)


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


def midpoint_energy_mj_for_file(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows:
        return None
    header = rows[0]

    # Support both naming variants.
    if "energy_per_packet_min_mj" in header and "energy_per_packet_max_mj" in header:
        i_min = header.index("energy_per_packet_min_mj")
        i_max = header.index("energy_per_packet_max_mj")
        scale = 1.0
    elif "energy_per_packet_j_min" in header and "energy_per_packet_j_max" in header:
        i_min = header.index("energy_per_packet_j_min")
        i_max = header.index("energy_per_packet_j_max")
        scale = 1000.0  # convert J -> mJ
    else:
        return None

    mids_raw = []
    for row in rows[1:]:
        if len(row) <= max(i_min, i_max):
            continue
        e_min = parse_float(row[i_min])
        e_max = parse_float(row[i_max])
        if e_min is None or e_max is None:
            continue
        mids_raw.append((e_min + e_max) / 2.0)

    if not mids_raw:
        return None

    # Some raw files use *_mj headers but store Joules (e.g. ~0.04).
    # Detect and correct by converting to mJ for plotting consistency.
    if scale == 1.0:
        sample = mids_raw[: min(20, len(mids_raw))]
        sample_avg = sum(sample) / len(sample)
        if sample_avg < 1.0:
            scale = 1000.0

    mids_mj = [m * scale for m in mids_raw]
    return sum(mids_mj) / len(mids_mj)


def midpoint_energy_sum_count_for_file(path):
    """Return (sum_mj, count) for weighted averaging over all distances. Returns None if no data."""
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
    mids_raw = []
    for row in rows[1:]:
        if len(row) <= max(i_min, i_max):
            continue
        e_min = parse_float(row[i_min])
        e_max = parse_float(row[i_max])
        if e_min is None or e_max is None:
            continue
        mids_raw.append((e_min + e_max) / 2.0)
    if not mids_raw:
        return None
    if scale == 1.0:
        sample = mids_raw[: min(20, len(mids_raw))]
        sample_avg = sum(sample) / len(sample)
        if sample_avg < 1.0:
            scale = 1000.0
    mids_mj = [m * scale for m in mids_raw]
    return sum(mids_mj), len(mids_mj)


def collect_data(data_root):
    # acc[(tp,bw,sf)] -> [file-level avg midpoint energies over distances]
    acc = defaultdict(list)
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
                v = midpoint_energy_mj_for_file(os.path.join(root, fn))
                if v is None:
                    continue
                acc[(tp, bw, sf)].append(v)

    # mean map
    mean_map = {}
    for key, vals in acc.items():
        mean_map[key] = sum(vals) / len(vals)
    return mean_map


# 1 Wh = 3600 J = 3,600,000 mJ
MJ_TO_WH = 1.0 / 3_600_000


def collect_total_energy_wh(data_root):
    """Total energy usage (Wh) per config, summed over all packets from all distances."""
    acc = defaultdict(float)
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
                result = midpoint_energy_sum_count_for_file(os.path.join(root, fn))
                if result is None:
                    continue
                total_mj, _ = result
                acc[(tp, bw, sf)] += total_mj
    return {k: v * MJ_TO_WH for k, v in acc.items() if v > 0}


def collect_data_over_all_distances(data_root):
    """Avg energy per packet for each config, aggregated over ALL packets from ALL distances (weighted avg)."""
    # acc[(tp,bw,sf)] -> (sum_mj, count)
    acc = defaultdict(lambda: (0.0, 0))
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
                result = midpoint_energy_sum_count_for_file(os.path.join(root, fn))
                if result is None:
                    continue
                s, n = result
                prev_s, prev_n = acc[(tp, bw, sf)]
                acc[(tp, bw, sf)] = (prev_s + s, prev_n + n)
    mean_map = {}
    for key, (total, count) in acc.items():
        if count > 0:
            mean_map[key] = total / count
    return mean_map


def parse_distance_weights(s):
    """Parse --distance-weights string. Format: '6.25:1,12.5:2,25:0.5' or preset 'inverse','inverse_sq','linear','peak75'.
    Returns dict {distance: weight}. Presets use distances from data; call with data_root for presets."""
    s = s.strip().lower()
    if s == "inverse":
        return "inverse"
    if s == "inverse_sq":
        return "inverse_sq"
    if s == "linear":
        return "linear"
    if s == "peak75":
        return "peak75"
    out = {}
    for part in s.split(","):
        part = part.strip()
        if ":" in part:
            d, w = part.split(":", 1)
            out[float(d.strip())] = float(w.strip())
    return out if out else None


def collect_data_weighted_by_distance(data_root, distance_weights):
    """Weighted avg of per-file means. distance_weights: dict {distance: weight} or preset str."""
    # First pass: get all distances
    distances = set()
    for dn in os.listdir(data_root):
        dpath = os.path.join(data_root, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        d = parse_distance(dn)
        if d is not None:
            distances.add(d)

    # Resolve preset
    if distance_weights == "inverse":
        weights_map = {d: 1.0 / d if d > 0 else 1.0 for d in distances}
    elif distance_weights == "inverse_sq":
        weights_map = {d: 1.0 / (d * d) if d > 0 else 1.0 for d in distances}
    elif distance_weights == "linear":
        weights_map = {d: d for d in distances}
    elif distance_weights == "peak75":
        # Gaussian-like: lowest at closest, peaks at ~75 m, curves down beyond
        sigma = 35.0
        peak = 75.0
        weights_map = {d: np.exp(-((d - peak) ** 2) / (2 * sigma**2)) for d in distances}
    elif isinstance(distance_weights, dict):
        weights_map = distance_weights
    else:
        weights_map = {}

    # acc[(tp,bw,sf)] -> (sum_mean_weighted, sum_weight)
    acc = defaultdict(lambda: (0.0, 0.0))
    for dn in sorted(os.listdir(data_root)):
        dpath = os.path.join(data_root, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        distance = parse_distance(dn)
        if distance is None:
            continue
        w = weights_map.get(distance, 1.0)
        for root, _, files in os.walk(dpath):
            for fn in files:
                cfg = parse_cfg(fn)
                if cfg is None:
                    continue
                sf, bw, tp = cfg
                if sf not in SF_VALUES or bw not in BW_VALUES or tp not in TP_VALUES:
                    continue
                v = midpoint_energy_mj_for_file(os.path.join(root, fn))
                if v is None:
                    continue
                prev_sum, prev_w = acc[(tp, bw, sf)]
                acc[(tp, bw, sf)] = (prev_sum + v * w, prev_w + w)
    mean_map = {}
    for key, (s, w) in acc.items():
        if w > 0:
            mean_map[key] = s / w
    return mean_map


def load_mean_map_from_csv(csv_path, energy_col="energy_mj"):
    """Load mean_map from CSV. Returns None if file does not exist."""
    if not os.path.isfile(csv_path):
        return None
    mean_map = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if energy_col not in row:
                return None
            tp = int(row["tp"])
            bw = int(row["bw"])
            sf = int(row["sf"])
            mean_map[(tp, bw, sf)] = float(row[energy_col])
    return mean_map


def save_mean_map_to_csv(mean_map, csv_path, energy_col="energy_mj"):
    """Save mean_map to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tp", "bw", "sf", energy_col])
        for (tp, bw, sf), val in sorted(mean_map.items()):
            writer.writerow([tp, bw, sf, val])
    print(f"Saved energy values to {csv_path}")


def plot(mean_map, output_png, cmap="gray", ylabel=None):
    if ylabel is None:
        ylabel = r"$\bar{E}_{\mathrm{pkt}}$ (mJ)"
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    # Stackplot: x = BW-TP pairs (sorted by total energy), stacks = SF
    configs = []
    for bw in BW_VALUES:
        for tp in TP_VALUES:
            sf_vals = []
            total = 0
            for sf in SF_VALUES:
                v = mean_map.get((tp, bw, sf))
                val = v if v is not None else 0
                sf_vals.append(val)
                total += val
            label = f"{bw/1000:.1f}".rstrip("0").rstrip(".") + "-TP" + str(tp)
            configs.append((total, label, sf_vals))
    configs.sort(key=lambda c: c[0])
    labels = [c[1] for c in configs]
    sf_arrays = [[c[2][i] for c in configs] for i in range(len(SF_VALUES))]
    n_configs = len(configs)
    any_data = any(sf_arrays[i][j] > 0 for i in range(len(SF_VALUES)) for j in range(n_configs))
    if not any_data:
        raise RuntimeError("No usable energy values found in dataset.")

    x = np.arange(n_configs)
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_TWO_COL)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(SF_VALUES)))
    ax.stackplot(x, *sf_arrays, labels=[f"SF{s}" for s in SF_VALUES], colors=colors)
    ax.set_xlabel("Config (BW-TP)", fontsize=IEEE_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=IEEE_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlim(0.0, n_configs-1)
    ax.legend(loc="upper left", fontsize=IEEE_FONTSIZE, ncol=3)
    ax.set_ylim(bottom=0)
    fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.22)
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    print(f"Saved plot: {output_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot avg midpoint packet energy heatmaps (BW vs SF) for each TP."
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join(WORKSPACE, "raw_test_data"),
        help="Dataset root to read (raw_test_data or dataset).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG base path (default: energy_bw_vs_sf_with_tp/raw_energy_minmax_gradient_by_tp).",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recompute from raw data (ignore cached CSV).",
    )
    parser.add_argument(
        "--mode",
        choices=["per-packet", "wh"],
        default="per-packet",
        help="per-packet: avg energy per packet (mJ); wh: total energy usage (Wh).",
    )
    args = parser.parse_args()
    if args.output is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        subdir = "energy_bw_vs_sf_with_tp"
        base_name = "dataset_energy_minmax_gradient_by_tp" if "dataset" in args.data_root else "raw_energy_minmax_gradient_by_tp"
        suffix = "_wh" if args.mode == "wh" else "_v3"
        args.output = os.path.join(WORKSPACE, "results", out_dir, subdir, f"{base_name}{suffix}.png")

    setup_plot_style()
    base = args.output
    if base.lower().endswith(".png"):
        base = base[:-4]
    csv_path = base + "_values.csv"
    if args.mode == "wh":
        collect_fn = collect_total_energy_wh
        energy_col = "energy_wh"
        ylabel = "Energy (Wh)"
    else:
        collect_fn = lambda dr: collect_data_weighted_by_distance(dr, "peak75")
        energy_col = "energy_mj"
        ylabel = None
    if args.recompute:
        mean_map = collect_fn(args.data_root)
        save_mean_map_to_csv(mean_map, csv_path, energy_col=energy_col)
    else:
        mean_map = load_mean_map_from_csv(csv_path, energy_col=energy_col)
        if mean_map is None:
            mean_map = collect_fn(args.data_root)
            save_mean_map_to_csv(mean_map, csv_path, energy_col=energy_col)
    plot(mean_map, f"{base}.png", cmap="viridis_r", ylabel=ylabel)


if __name__ == "__main__":
    main()
