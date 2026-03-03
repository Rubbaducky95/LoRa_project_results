"""
Plot RSSI (avg, raw) vs various x-axis metrics.
Generates multiple plots to show impact on RSSI from distance, SF, BW, TP, energy, throughput, etc.
"""
import argparse
import csv
import os
import re
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox, Affine2D
from mpl_toolkits.mplot3d import Axes3D

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


def read_file_metrics(path):
    """
    Return dict with: rssi_avg, distance, sf, bw, tp, energy_mj, throughput_bps, energy_per_bit_uj, snr_db.
    Returns None if insufficient data.
    """
    rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
    if not rows:
        return None
    header = rows[0]
    if "rssi" not in header:
        return None
    i_rssi = header.index("rssi")
    i_emin = header.index("energy_per_packet_min_mj") if "energy_per_packet_min_mj" in header else None
    i_emax = header.index("energy_per_packet_max_mj") if "energy_per_packet_max_mj" in header else None
    if i_emin is None and "energy_per_packet_j_min" in header:
        i_emin = header.index("energy_per_packet_j_min")
        i_emax = header.index("energy_per_packet_j_max")
        energy_scale = 1000.0
    else:
        energy_scale = 1.0
    i_time = header.index("time_since_transmission_init_ms") if "time_since_transmission_init_ms" in header else None
    i_payload = header.index("payload_size_bytes") if "payload_size_bytes" in header else None
    i_snr = header.index("snr_db") if "snr_db" in header else None

    rssi_vals = []
    energy_mids = []
    times_ms = []
    payload_bytes = []
    snr_vals = []
    for row in rows[1:]:
        if len(row) <= i_rssi:
            continue
        r = parse_float(row[i_rssi])
        if r is not None:
            rssi_vals.append(r)
        if i_emin is not None and i_emax is not None and len(row) > max(i_emin, i_emax):
            e1, e2 = parse_float(row[i_emin]), parse_float(row[i_emax])
            if e1 is not None and e2 is not None:
                energy_mids.append((e1 + e2) / 2.0 * energy_scale)
        if i_time is not None and len(row) > i_time:
            t = parse_float(row[i_time])
            if t is not None:
                times_ms.append(t)
        if i_payload is not None and len(row) > i_payload:
            p = parse_float(row[i_payload])
            if p is not None and p > 0:
                payload_bytes.append(int(p))
        if i_snr is not None and len(row) > i_snr:
            s = parse_float(row[i_snr])
            if s is not None:
                snr_vals.append(s)

    if not rssi_vals:
        return None

    if energy_scale == 1.0 and energy_mids:
        sample = energy_mids[: min(20, len(energy_mids))]
        if sum(sample) / len(sample) < 1.0:
            energy_mids = [e * 1000.0 for e in energy_mids]

    rssi_avg = sum(rssi_vals) / len(rssi_vals)
    energy_mj = sum(energy_mids) / len(energy_mids) if energy_mids else None
    count = len(rssi_vals)
    duration_s = (max(times_ms) - min(times_ms)) / 1000.0 if len(times_ms) >= 2 else 0.0
    payload_b = int(sum(payload_bytes) / len(payload_bytes)) if payload_bytes else 37
    total_bits = count * payload_b * 8
    throughput_bps = total_bits / duration_s if duration_s > 0 else None
    energy_per_bit_uj = (sum(energy_mids) * 1000.0 / total_bits) if energy_mids and total_bits > 0 else None
    snr_avg = sum(snr_vals) / len(snr_vals) if snr_vals else None

    return {
        "rssi_avg": rssi_avg,
        "energy_mj": energy_mj,
        "throughput_bps": throughput_bps,
        "energy_per_bit_uj": energy_per_bit_uj,
        "snr_db": snr_avg,
        "count": count,
    }


def collect_rssi_data(data_root):
    """
    Per file: (distance, sf, bw, tp) -> metrics dict.
    Returns list of (distance, sf, bw, tp, metrics).
    """
    records = []
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
                if sf not in SF_VALUES or bw not in BW_VALUES or tp not in TP_VALUES:
                    continue
                metrics = read_file_metrics(os.path.join(root, fn))
                if metrics is None:
                    continue
                records.append((distance, sf, bw, tp, metrics))
    return records


def plot_rssi_vs_x(records, x_key, x_label, output_png, x_unit=""):
    """Scatter: RSSI (avg) vs x_key. Color by SF."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    points = []
    for distance, sf, bw, tp, m in records:
        x_val = None
        if x_key == "distance":
            x_val = distance
        elif x_key == "sf":
            x_val = sf
        elif x_key == "bw":
            x_val = bw / 1000.0
        elif x_key == "tp":
            x_val = tp
        elif x_key == "energy_mj":
            x_val = m.get("energy_mj")
        elif x_key == "throughput_bps":
            x_val = m.get("throughput_bps")
        elif x_key == "energy_per_bit_uj":
            x_val = m.get("energy_per_bit_uj")
        elif x_key == "snr_db":
            x_val = m.get("snr_db")
        if x_val is None:
            continue
        points.append((sf, bw, x_val, m["rssi_avg"]))

    if not points:
        return

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_ONE_COL)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(SF_VALUES)))
    for sf_idx, sf in enumerate(SF_VALUES):
        pts = [p for p in points if p[0] == sf]
        if not pts:
            continue
        x_vals = [p[2] for p in pts]
        y_vals = [p[3] for p in pts]
        ax.scatter(x_vals, y_vals, c=[colors[sf_idx]], label=f"SF{sf}", s=25, marker="o", edgecolors="k", linewidths=0.3)
    ax.set_xlabel(x_label + (f" ({x_unit})" if x_unit else ""), fontsize=IEEE_FONTSIZE)
    ax.set_ylabel(r"RSSI (avg) (dBm)", fontsize=IEEE_FONTSIZE)
    ax.legend(loc="best", fontsize=IEEE_FONTSIZE, ncol=2)
    ax.set_xlim(left=min(p[2] for p in points) * 0.98 if min(p[2] for p in points) > 0 else None)
    fig.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.12)
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


def _fmt_bw(b):
    return f"{b:.1f}".rstrip("0").rstrip(".") if b != int(b) else str(int(b))


def _get_axis_val(axis, distance, sf, bw, tp):
    if axis == "distance":
        return distance
    if axis == "sf":
        return sf
    if axis == "bw":
        return bw / 1000.0
    if axis == "tp":
        return tp
    return None


def plot_rssi_3d(records, output_png, x_axis, y_axis, z_axis, invert_x=False, invert_y=False):
    """3D scatter: x_axis, y_axis, z_axis as axes, color=RSSI (avg). Aggregates over the 4th dimension."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    agg = defaultdict(list)
    for distance, sf, bw, tp, m in records:
        key = (_get_axis_val(x_axis, distance, sf, bw, tp),
               _get_axis_val(y_axis, distance, sf, bw, tp),
               _get_axis_val(z_axis, distance, sf, bw, tp))
        if None in key:
            continue
        agg[key].append(m["rssi_avg"])
    points = []
    for (xv, yv, zv), rssi_list in agg.items():
        points.append({x_axis: xv, y_axis: yv, z_axis: zv, "rssi": sum(rssi_list) / len(rssi_list)})
    if not points:
        return
    x_vals = np.array([p[x_axis] for p in points])
    y_vals = np.array([p[y_axis] for p in points])
    z_vals = np.array([p[z_axis] for p in points])
    rssi_vals = np.array([p["rssi"] for p in points])
    fig = plt.figure(figsize=FIGSIZE_TWO_COL)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x_vals, y_vals, z_vals, c=rssi_vals, cmap="viridis", s=50, edgecolors="k", linewidths=0.3)
    labels = {"sf": "SF", "bw": "BW (kHz)", "tp": "TP (dBm)", "distance": "Distance (m)"}
    ax.set_xlabel(labels.get(x_axis, x_axis), fontsize=IEEE_FONTSIZE)
    ax.set_ylabel(labels.get(y_axis, y_axis), fontsize=IEEE_FONTSIZE)
    ax.set_zlabel(labels.get(z_axis, z_axis), fontsize=IEEE_FONTSIZE)
    bw_khz = sorted(set(bw / 1000.0 for bw in BW_VALUES))
    distances = sorted(set(d for d, _, _, _, _ in records))
    for axis, key in [("x", x_axis), ("y", y_axis), ("z", z_axis)]:
        set_ticks = getattr(ax, f"set_{axis}ticks")
        set_ticklabels = getattr(ax, f"set_{axis}ticklabels")
        if key == "sf":
            set_ticks(SF_VALUES)
            set_ticklabels([str(s) for s in SF_VALUES])
        elif key == "bw":
            set_ticks(bw_khz)
            set_ticklabels([_fmt_bw(b) for b in bw_khz])
        elif key == "tp":
            set_ticks(TP_VALUES)
            set_ticklabels([str(t) for t in TP_VALUES])
        elif key == "distance":
            set_ticks(distances)
            set_ticklabels([str(int(d)) if d == int(d) else f"{d:.1f}" for d in distances])
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, location="left", pad=0.001)
    cbar.set_label(r"RSSI (avg) (dBm)", fontsize=IEEE_FONTSIZE, rotation=90, labelpad=0.001)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.tick_params(axis="y", which="both", left=True, right=False, labelleft=True, labelright=False)
    fig.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.05)
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


DISTANCE_TICK_INTERVAL = 25.0


def plot_rssi_3d_combined(records, output_png):
    """Three 3D subplots in one 2-column figure: A) BW-distance-SF, B) TP-distance-SF, C) TP-distance-BW."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    distances = sorted(set(d for d, _, _, _, _ in records))
    d_min, d_max = min(distances), max(distances)
    dist_ticks = [DISTANCE_TICK_INTERVAL * i for i in range(1, int(d_max / DISTANCE_TICK_INTERVAL) + 1) if d_min <= DISTANCE_TICK_INTERVAL * i <= d_max]
    bw_khz = sorted(set(bw / 1000.0 for bw in BW_VALUES))
    labels = {"sf": "SF", "bw": "BW (kHz)", "tp": "TP (dBm)", "distance": "Distance (m)"}

    def agg_3d(x_axis, y_axis, z_axis):
        a = defaultdict(list)
        for distance, sf, bw, tp, m in records:
            key = (_get_axis_val(x_axis, distance, sf, bw, tp),
                   _get_axis_val(y_axis, distance, sf, bw, tp),
                   _get_axis_val(z_axis, distance, sf, bw, tp))
            if None in key:
                continue
            a[key].append(m["rssi_avg"])
        return [{x_axis: xv, y_axis: yv, z_axis: zv, "rssi": sum(v) / len(v)} for (xv, yv, zv), v in a.items()]

    configs = [
        ("bw", "distance", "sf", True, False),
        ("tp", "distance", "sf", True, False),
        ("tp", "distance", "bw", True, False),
    ]
    fig = plt.figure(figsize=(FIGSIZE_TWO_COL[0], 4.5))
    rssi_all = []
    for x_axis, y_axis, z_axis, _, _ in configs:
        pts = agg_3d(x_axis, y_axis, z_axis)
        rssi_all.extend(p["rssi"] for p in pts)
    rssi_min, rssi_max = min(rssi_all), max(rssi_all) if rssi_all else (-100, -40)

    scatter_handles = []
    for idx, (x_axis, y_axis, z_axis, inv_x, inv_y) in enumerate(configs):
        pts = agg_3d(x_axis, y_axis, z_axis)
        if not pts:
            continue
        x_vals = np.array([p[x_axis] for p in pts])
        y_vals = np.array([p[y_axis] for p in pts])
        z_vals = np.array([p[z_axis] for p in pts])
        if x_axis == "bw" or y_axis == "bw" or z_axis == "bw":
            for axis, key in [("x", x_axis), ("y", y_axis), ("z", z_axis)]:
                if key == "bw":
                    if axis == "x":
                        x_vals = np.log10(np.maximum(x_vals, 1))
                    elif axis == "y":
                        y_vals = np.log10(np.maximum(y_vals, 1))
                    else:
                        z_vals = np.log10(np.maximum(z_vals, 1))
                    break
        rssi_vals = np.array([p["rssi"] for p in pts])
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        sc = ax.scatter(x_vals, y_vals, z_vals, c=rssi_vals, cmap="viridis", s=35, edgecolors="k", linewidths=0.2, vmin=rssi_min, vmax=rssi_max)
        scatter_handles.append((sc, ax, x_axis, y_axis, z_axis))
        ax.set_xlabel(labels[x_axis], fontsize=IEEE_FONTSIZE)
        ax.set_ylabel(labels[y_axis], fontsize=IEEE_FONTSIZE)
        ax.set_zlabel(labels[z_axis], fontsize=IEEE_FONTSIZE)
        for axis, key in [("x", x_axis), ("y", y_axis), ("z", z_axis)]:
            set_ticks = getattr(ax, f"set_{axis}ticks")
            set_ticklabels = getattr(ax, f"set_{axis}ticklabels")
            if key == "sf":
                set_ticks(SF_VALUES)
                set_ticklabels([str(s) for s in SF_VALUES])
            elif key == "bw":
                log_bw = np.log10(np.array(bw_khz))
                set_ticks(log_bw)
                set_ticklabels([_fmt_bw(b) for b in bw_khz])
            elif key == "tp":
                set_ticks(TP_VALUES)
                set_ticklabels([str(t) for t in TP_VALUES])
            elif key == "distance":
                set_ticks(dist_ticks)
                set_ticklabels([f"{int(d)}" for d in dist_ticks])
        if inv_x:
            ax.invert_xaxis()
        if inv_y:
            ax.invert_yaxis()

    fig.subplots_adjust(left=0.05, right=0.95, top=0.82, bottom=0.05, wspace=0.25)
    cbars = []
    for sc, ax, _, _, _ in scatter_handles:
        cbar = fig.colorbar(sc, ax=ax, location="top", orientation="horizontal", shrink=0.6, pad=0.08, aspect=30, fraction=0.04)
        cbar.ax.invert_xaxis()
        cbar.set_label(r"RSSI (avg) (dBm)", fontsize=IEEE_FONTSIZE - 1)
        cbar.ax.tick_params(labelsize=IEEE_FONTSIZE - 2, rotation=45)
        cbars.append(cbar)
    fig.canvas.draw()
    for cbar in cbars:
        cbar_ax = cbar.ax
        pos = cbar_ax.get_position(original=False)
        cx, cy = pos.x0 + pos.width / 2, pos.y0 + pos.height / 2
        t = Affine2D().translate(-cx, -cy).rotate_deg(45).translate(cx, cy)
        orig_bbox = Bbox.from_bounds(pos.x0, pos.y0, pos.width, pos.height)

        def make_locator(bbox, trans):
            def locator(axes, renderer):
                return trans.transform_bbox(bbox)
            return locator

        cbar_ax.set_axes_locator(make_locator(orig_bbox, t))
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Plot RSSI (avg) vs various x-axis metrics.")
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
    args = parser.parse_args()

    if args.output_dir is None:
        out_dir = "dataset_plots" if "dataset" in args.data_root else "raw_test_data_plots"
        args.output_dir = os.path.join(WORKSPACE, "results", out_dir, "rssi")

    setup_plot_style()
    records = collect_rssi_data(args.data_root)
    if not records:
        raise RuntimeError("No RSSI data found.")

    plot_rssi_3d_combined(records, os.path.join(args.output_dir, "raw_rssi_3d_combined.png"))


if __name__ == "__main__":
    main()
