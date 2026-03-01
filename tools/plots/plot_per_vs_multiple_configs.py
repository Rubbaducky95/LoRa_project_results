"""
Plot PER vs RSSI, Tx interval, and distance. Use --config to choose which parameters
(sf, bw, tp) to separate by; others are averaged. E.g. --config sf-bw plots all SF,BW combos.
"""
import argparse
import csv
import math
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
DATA_ROOT = os.path.join(WORKSPACE, "raw_test_data")

BW_VALUES = [62500, 125000, 250000, 500000]
TP_VALUES = [2, 12, 22]
SF_VALUES = [7, 8, 9, 10, 11, 12]
HEX_RE = re.compile(r"^[0-9A-F]+$")

MARKERS = "o s ^ D v < > p h * P X".split()
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


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


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    return tuple(map(int, m.groups())) if m else None


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def get_file_order():
    """Yield (sf, bw, tp) in test order: SF, BW, TP."""
    for sf in SF_VALUES:
        for bw in BW_VALUES:
            for tp in TP_VALUES:
                yield (sf, bw, tp)


def find_file_in_dist(dist_path, sf, bw, tp):
    """Find CSV path for given config within a distance folder."""
    for sub in os.listdir(dist_path):
        spath = os.path.join(dist_path, sub)
        if not os.path.isdir(spath):
            continue
        for fn in os.listdir(spath):
            cfg = parse_cfg(fn)
            if cfg and cfg == (sf, bw, tp):
                return os.path.join(spath, fn)
    return None


def payload_is_valid(payload):
    if payload == "PACKET_LOST":
        return False
    parts = (payload or "").strip('"').split(",")
    for part in parts:
        if part and HEX_RE.match(part) is None:
            return False
    return True


def payload_first_value_ms(payload):
    """Extract first value from payload as ms since boot (sender timestamp). Returns None if invalid."""
    if not payload or payload == "PACKET_LOST" or str(payload).strip().startswith("CFG "):
        return None
    parts = str(payload).strip('"').split(",")
    if not parts or not parts[0].strip():
        return None
    try:
        return float(parts[0].strip())
    except (ValueError, TypeError):
        return None


def file_metrics(path, bw_hz):
    """Returns (per, avg_rssi, avg_tx_interval_ms, packets_per_minute, avg_energy_mj) or None.
    Does NOT include time_min - use per-packet time_since_boot_ms for time plots."""
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "payload" not in rows[0]:
        return None
    header = rows[0]
    payload_idx = header.index("payload")
    rssi_idx = header.index("rssi_corrected") if "rssi_corrected" in header else (header.index("rssi") if "rssi" in header else None)
    tx_idx = header.index("tx_interval_ms") if "tx_interval_ms" in header else None
    ts_idx = header.index("timestamp") if "timestamp" in header else None
    e_min_idx = header.index("energy_per_packet_min_mj") if "energy_per_packet_min_mj" in header else None
    e_max_idx = header.index("energy_per_packet_max_mj") if "energy_per_packet_max_mj" in header else None

    total = 0
    lost = 0
    rssi_vals = []
    tx_vals = []
    timestamps = []
    energy_mids = []
    for r in rows[1:]:
        if len(r) <= payload_idx:
            continue
        if str(r[payload_idx]).strip().startswith("CFG "):
            continue
        total += 1
        if not payload_is_valid(r[payload_idx]):
            lost += 1
        if rssi_idx is not None and len(r) > rssi_idx and r[rssi_idx]:
            try:
                rssi_vals.append(float(r[rssi_idx]))
            except (ValueError, TypeError):
                pass
        if tx_idx is not None and len(r) > tx_idx and r[tx_idx]:
            try:
                tx_vals.append(float(r[tx_idx]))
            except (ValueError, TypeError):
                pass
        if ts_idx is not None and len(r) > ts_idx and r[ts_idx]:
            try:
                timestamps.append(datetime.fromisoformat(r[ts_idx].replace("Z", "+00:00")))
            except Exception:
                pass
        if e_min_idx is not None and e_max_idx is not None and len(r) > max(e_min_idx, e_max_idx):
            try:
                e_min = float(r[e_min_idx])
                e_max = float(r[e_max_idx])
                energy_mids.append((e_min + e_max) / 2.0)
            except (ValueError, TypeError):
                pass
    if total == 0:
        return None
    per = lost / total

    avg_rssi = np.mean(rssi_vals) if rssi_vals else None
    avg_tx = np.mean(tx_vals) if tx_vals else None
    if avg_tx and avg_tx > 0:
        ppm = 60000.0 / avg_tx
    elif len(timestamps) >= 2:
        duration_s = (max(timestamps) - min(timestamps)).total_seconds()
        ppm = 60.0 * total / duration_s if duration_s > 0 else None
    else:
        ppm = None
    avg_energy = np.mean(energy_mids) if energy_mids else None

    return (per, avg_rssi, avg_tx, ppm, avg_energy)


def file_packets(path, distance, sf, bw, tp):
    """Yield (time_since_boot_ms, lost, rssi, energy, tx_interval_ms) for each packet."""
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows or "payload" not in rows[0]:
        return
    header = rows[0]
    payload_idx = header.index("payload")
    time_idx = header.index("time_since_boot_ms") if "time_since_boot_ms" in header else None
    tx_idx = header.index("tx_interval_ms") if "tx_interval_ms" in header else None
    rssi_idx = header.index("rssi_corrected") if "rssi_corrected" in header else (header.index("rssi") if "rssi" in header else None)
    e_min_idx = header.index("energy_per_packet_min_mj") if "energy_per_packet_min_mj" in header else None
    e_max_idx = header.index("energy_per_packet_max_mj") if "energy_per_packet_max_mj" in header else None
    if time_idx is None:
        return
    for r in rows[1:]:
        if len(r) <= payload_idx:
            continue
        if str(r[payload_idx]).strip().startswith("CFG "):
            continue
        t_ms = None
        if time_idx is not None and len(r) > time_idx and r[time_idx]:
            try:
                t_ms = float(r[time_idx])
            except (ValueError, TypeError):
                pass
        if t_ms is None:
            continue
        tx_ms = 1550.0
        if tx_idx is not None and len(r) > tx_idx and r[tx_idx]:
            try:
                tx_ms = float(r[tx_idx])
            except (ValueError, TypeError):
                pass
        lost = 0 if payload_is_valid(r[payload_idx]) else 1
        rssi = None
        if rssi_idx is not None and len(r) > rssi_idx and r[rssi_idx]:
            try:
                rssi = float(r[rssi_idx])
            except (ValueError, TypeError):
                pass
        energy = None
        if e_min_idx is not None and e_max_idx is not None and len(r) > max(e_min_idx, e_max_idx):
            try:
                energy = (float(r[e_min_idx]) + float(r[e_max_idx])) / 2.0
            except (ValueError, TypeError):
                pass
        yield (t_ms, lost, rssi, energy, tx_ms)


CONFIG_DIMS = {"sf": SF_VALUES, "bw": BW_VALUES, "tp": TP_VALUES, "distance": None}


def parse_config_arg(config_str):
    """Parse --config string like 'sf-bw' into ordered list of dimension names. Raises if invalid."""
    parts = [p.strip().lower() for p in config_str.split("-") if p.strip()]
    seen = set()
    dims = []
    for p in parts:
        if p not in CONFIG_DIMS:
            raise ValueError(f"Unknown config dimension '{p}'. Valid: sf, bw, tp, distance")
        if p in seen:
            raise ValueError(f"Duplicate dimension '{p}' in config")
        seen.add(p)
        dims.append(p)
    if not dims:
        raise ValueError("Config must specify at least one dimension (sf, bw, tp, distance)")
    if "distance" in dims and len(dims) > 1:
        raise ValueError("Distance must be used alone (--config distance), not combined with sf/bw/tp")
    if set(dims) == {"sf", "bw", "tp"}:
        raise ValueError("sf-bw-tp creates too many configurations; use sf-bw, sf-tp, or tp-bw instead")
    return dims


def config_key_from_row(dims, sf, bw, tp, distance=None):
    """Build config key tuple from row values, in order of dims."""
    lookup = {"sf": sf, "bw": bw, "tp": tp, "distance": distance}
    return tuple(lookup[d] for d in dims)


def parse_filter_spec(spec):
    """Parse filter spec like 'sf10,tp22,distance93.75' into dict. Returns None if invalid."""
    out = {}
    for part in spec.lower().split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^sf(\d+)$", part)
        if m:
            out["sf"] = int(m.group(1))
            continue
        m = re.match(r"^bw(\d+)$", part)
        if m:
            out["bw"] = int(m.group(1))
            continue
        m = re.match(r"^tp(\d+)$", part)
        if m:
            out["tp"] = int(m.group(1))
            continue
        m = re.match(r"^distance([\d.]+)$", part)
        if m:
            out["distance"] = float(m.group(1))
            continue
    return out if out else None


def config_matches_filter(sf, bw, tp, distance, filter_spec):
    """True if (sf,bw,tp,distance) matches the filter spec (should be excluded)."""
    for k, v in filter_spec.items():
        if k == "sf" and sf != v:
            return False
        if k == "bw" and bw != v:
            return False
        if k == "tp" and tp != v:
            return False
        if k == "distance" and distance is not None and abs(distance - v) > 0.01:
            return False
    return True


def fit_curve_log(xs, ys, degree=2):
    """Fit polynomial to log10(y) vs x. Returns (x_smooth, y_smooth) or None if insufficient points."""
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < degree + 2 or np.any(ys <= 0):
        return None
    log_ys = np.log10(ys)
    try:
        coeffs = np.polyfit(xs, log_ys, degree)
        x_smooth = np.linspace(xs.min(), xs.max(), 100)
        y_smooth = 10 ** np.polyval(coeffs, x_smooth)
        return x_smooth, y_smooth
    except (np.linalg.LinAlgError, ValueError):
        return None


def format_config_label(dims, key):
    """Format config key for legend, e.g. (7, 125000) -> 'SF7 BW125' for dims=['sf','bw']."""
    parts = []
    for i, d in enumerate(dims):
        v = key[i] if isinstance(key, (tuple, list)) else key
        if d == "sf":
            parts.append(f"SF{v}")
        elif d == "bw":
            parts.append(f"BW{v // 1000}")
        elif d == "tp":
            parts.append(f"TP{v}")
        elif d == "distance":
            parts.append(f"{v}m")
    return " ".join(parts)


def collect_by_config(data_root, dims, filters=None):
    """
    Returns {config_key: [(rssi, per, distance, energy_mj), ...]}.
    config_key is a tuple of values for the dimensions in dims (e.g. (sf, bw) or (sf, distance)).
    When distance in dims: one point per (sf,bw,tp,distance). Otherwise: one point per distance, averaged.
    """
    filters = filters or []
    dist_in_dims = "distance" in dims
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
                if any(config_matches_filter(sf, bw, tp, distance, f) for f in filters):
                    continue
                m = file_metrics(os.path.join(root, fn), bw)
                if m is None:
                    continue
                per, rssi, _, _, energy = m
                key = config_key_from_row(dims, sf, bw, tp, distance)
                by_cfg[key].append((distance, rssi, per, energy))

    result = {}
    for key, pts in by_cfg.items():
        if dist_in_dims:
            vals = pts
            rssis = [v[1] for v in vals if v[1] is not None]
            pers = [v[2] for v in vals]
            energies = [v[3] for v in vals if v[3] is not None]
            dist_val = pts[0][0] if pts else 0
            result[key] = [(
                np.mean(rssis) if rssis else None,
                np.mean(pers),
                dist_val,
                np.mean(energies) if energies else None,
            )]
        else:
            by_dist = defaultdict(list)
            for d, rssi, per, energy in pts:
                by_dist[d].append((rssi, per, energy))
            out = []
            for d in sorted(by_dist.keys()):
                vals = by_dist[d]
                rssis = [v[0] for v in vals if v[0] is not None]
                pers = [v[1] for v in vals]
                energies = [v[2] for v in vals if v[2] is not None]
                out.append((
                    np.mean(rssis) if rssis else None,
                    np.mean(pers),
                    d,
                    np.mean(energies) if energies else None,
                ))
            result[key] = out
    return result


def collect_packets_for_time_plot(data_root, dims, filters=None):
    """
    Returns {config_key: {minute_bucket: (lost_count, total_count)}}.
    Uses internal timer: first value = 0, count up with tx_interval. Reset on distance change or dip.
    """
    filters = filters or []
    by_cfg = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [lost, total]
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
            key = config_key_from_row(dims, sf, bw, tp, distance)
            for raw_ms, lost, _, _, tx_ms in file_packets(path, distance, sf, bw, tp):
                if prev_raw_ms is not None and raw_ms < prev_raw_ms:
                    timer_ms = 0
                min_bucket = int(np.floor(timer_ms / 60000.0))
                by_cfg[key][min_bucket][0] += lost
                by_cfg[key][min_bucket][1] += 1
                timer_ms += tx_ms
                prev_raw_ms = raw_ms
    return dict(by_cfg)


def main():
    parser = argparse.ArgumentParser(
        description="PER vs RSSI, time, distance, energy. Use --config to choose which params to separate by."
    )
    parser.add_argument("--data-root", default=DATA_ROOT, help="Dataset root.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument(
        "--config",
        default="sf",
        help="Dimensions to separate by (others averaged). E.g. sf-bw, tp-bw. Use 'distance' alone for per-distance. Valid: sf, bw, tp, distance.",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Generate plots for all config combinations (sf, bw, tp, sf-bw, sf-tp, tp-bw, sf-bw-tp).",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="SPEC",
        help="Exclude configs, e.g. sf10,tp22,distance93.75. Can be repeated.",
    )
    args = parser.parse_args()

    if args.all_configs:
        config_list = ["sf", "bw", "tp", "distance", "sf-bw", "sf-tp", "tp-bw"]
        for cfg in config_list:
            args.config = cfg
            main_inner(args)
        return

    main_inner(args)


def main_inner(args):
    if args.output_dir is None:
        base = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        args.output_dir = os.path.join(WORKSPACE, "results", base)

    setup_plot_style()
    os.makedirs(args.output_dir, exist_ok=True)

    dims = parse_config_arg(args.config)
    filters = [parse_filter_spec(f) for f in (args.filter or [])]
    filters = [f for f in filters if f is not None]
    by_cfg = collect_by_config(args.data_root, dims, filters)
    time_by_cfg = collect_packets_for_time_plot(args.data_root, dims, filters)
    configs = sorted(by_cfg.keys())
    color_map = {c: COLORS[i % len(COLORS)] for i, c in enumerate(configs)}
    marker_map = {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(configs)}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Point tuple: (rssi, per, distance, energy_mj)
    # p[0]=rssi, p[1]=per, p[2]=distance, p[3]=energy

    # PER vs RSSI (log scale for PER)
    ax = axes[0, 0]
    for cfg in configs:
        pts = by_cfg[cfg]
        xs = [p[0] for p in pts if p[0] is not None and p[1] > 0]
        ys = [p[1] * 100 for p in pts if p[0] is not None and p[1] > 0]
        if not xs:
            continue
        order = np.argsort(xs)
        xs, ys = np.array(xs)[order], np.array(ys)[order]
        ax.scatter(xs, ys, s=40, c=[color_map[cfg]], marker=marker_map[cfg], alpha=0.9, edgecolors="black", linewidths=0.5)
    ax.set_xlabel("RSSI (dBm)")
    ax.set_ylabel("PER (%)")
    ax.set_title("PER vs RSSI")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(0.01, 100)

    # PER vs time (log scale for PER) - per-packet time_since_boot_ms, bucketed by minute
    ax = axes[0, 1]
    for cfg in configs:
        buckets = time_by_cfg.get(cfg, {})
        xs = []
        ys = []
        for min_bucket in sorted(buckets.keys()):
            lost, total = buckets[min_bucket]
            if total > 0:
                per_pct = 100.0 * lost / total
                if per_pct > 0:
                    xs.append(min_bucket)
                    ys.append(per_pct)
        if not xs:
            continue
        ax.scatter(xs, ys, s=40, c=[color_map[cfg]], marker=marker_map[cfg], alpha=0.9, edgecolors="black", linewidths=0.5)
        curve = fit_curve_log(np.array(xs), np.array(ys))
        if curve is not None:
            ax.plot(curve[0], curve[1], color=color_map[cfg], linestyle="-", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Time (minutes, internal timer from first packet)")
    ax.set_ylabel("PER (%)")
    ax.set_title("PER vs time")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(0.01, 100)

    # PER vs distance (log scale for PER)
    ax = axes[1, 0]
    for cfg in configs:
        pts = by_cfg[cfg]
        xs = [p[2] for p in pts if p[1] > 0]
        ys = [p[1] * 100 for p in pts if p[1] > 0]
        if not xs:
            continue
        order = np.argsort(xs)
        xs, ys = np.array(xs)[order], np.array(ys)[order]
        ax.scatter(xs, ys, s=40, c=[color_map[cfg]], marker=marker_map[cfg], alpha=0.9, edgecolors="black", linewidths=0.5)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("PER (%)")
    ax.set_title("PER vs distance")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(0.01, 100)

    # PER vs energy (log scale for PER) - single gray fit over all configs
    ax = axes[1, 1]
    all_xs, all_ys = [], []
    for cfg in configs:
        pts = by_cfg[cfg]
        xs = [p[3] for p in pts if p[3] is not None and p[1] > 0]
        ys = [p[1] * 100 for p in pts if p[3] is not None and p[1] > 0]
        if not xs:
            continue
        order = np.argsort(xs)
        xs, ys = np.array(xs)[order], np.array(ys)[order]
        ax.scatter(xs, ys, s=40, c=[color_map[cfg]], marker=marker_map[cfg], alpha=0.9, edgecolors="black", linewidths=0.5)
        all_xs.extend(xs.tolist())
        all_ys.extend(ys.tolist())
    if all_xs:
        order = np.argsort(all_xs)
        all_xs = np.array(all_xs)[order]
        all_ys = np.array(all_ys)[order]
        curve = fit_curve_log(all_xs, all_ys)
        if curve is not None:
            ax.plot(curve[0], curve[1], color="gray", linestyle="-", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Energy per packet (mJ)")
    ax.set_ylabel("PER (%)")
    ax.set_title("PER vs energy")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(0.01, 100)

    handles = [
        plt.Line2D([0], [0], marker=marker_map[c], color="w", markerfacecolor=color_map[c], markersize=8, label=format_config_label(dims, c))
        for c in configs
    ]
    ncol = min(len(configs), 8)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=7, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout()
    prefix = "dataset" if "dataset" in args.data_root else "raw"
    cfg_suffix = "_".join(dims)
    out_path = os.path.join(args.output_dir, f"{prefix}_per_vs_multiple_configs_{cfg_suffix}.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {prefix}_per_vs_multiple_configs_{cfg_suffix}.png")


if __name__ == "__main__":
    main()
