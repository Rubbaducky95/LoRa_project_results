import csv
import math
import os
import random
import re
import statistics
from collections import defaultdict
from datetime import datetime, timedelta


ROOT = r"C:\Users\ruben\Documents\LoRa Project\test_data2"
TARGET_DISTANCE = 100.0

SF_VALUES = [7, 8, 9, 10, 11, 12]
BW_VALUES = [62500, 125000, 250000, 500000]
TP_VALUES = [2, 12, 22]
EXCLUDED = {
    (11, 62500, 2),
    (11, 62500, 12),
    (11, 62500, 22),
    (12, 62500, 2),
    (12, 62500, 12),
    (12, 62500, 22),
}


def expected_configs():
    return [
        (sf, bw, tp)
        for sf in SF_VALUES
        for bw in BW_VALUES
        for tp in TP_VALUES
        if (sf, bw, tp) not in EXCLUDED
    ]


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.reader(f))


def collect_models():
    # Per config: distance -> (rssi_mean, rssi_std, time_mean, time_std)
    per_cfg = defaultdict(list)
    # TP-invariant time model: (sf,bw) -> distance -> time mean/std
    per_sfbw = defaultdict(list)

    for dn in sorted(os.listdir(ROOT)):
        folder = os.path.join(ROOT, dn)
        if not (os.path.isdir(folder) and dn.startswith("distance_")):
            continue
        dist = parse_distance(dn)
        if dist is None or abs(dist - TARGET_DISTANCE) < 1e-6:
            continue

        for fn in os.listdir(folder):
            cfg = parse_cfg(fn)
            if cfg is None:
                continue
            sf, bw, tp = cfg
            if cfg in EXCLUDED or tp not in TP_VALUES:
                continue

            rows = read_rows(os.path.join(folder, fn))
            if len(rows) < 3:
                continue
            rssi_vals = []
            time_vals = []
            for r in rows[2:]:
                if len(r) < 5:
                    continue
                try:
                    if r[1]:
                        rssi_vals.append(float(r[1]))
                except Exception:
                    pass
                try:
                    if r[4]:
                        time_vals.append(float(r[4]))
                except Exception:
                    pass
            if not rssi_vals or not time_vals:
                continue

            r_mean = statistics.mean(rssi_vals)
            r_std = statistics.stdev(rssi_vals) if len(rssi_vals) > 1 else 1.5
            t_mean = statistics.mean(time_vals)
            t_std = statistics.stdev(time_vals) if len(time_vals) > 1 else max(15.0, t_mean * 0.03)

            per_cfg[cfg].append((dist, r_mean, r_std, t_mean, t_std))
            per_sfbw[(sf, bw)].append((dist, t_mean, t_std))

    for k in list(per_cfg.keys()):
        per_cfg[k].sort(key=lambda x: x[0])

    # Merge TP variants for time model by same distance
    merged = {}
    for key, values in per_sfbw.items():
        bucket = defaultdict(list)
        for d, t_mean, t_std in values:
            bucket[d].append((t_mean, t_std))
        rows = []
        for d in sorted(bucket.keys()):
            t_means = [x[0] for x in bucket[d]]
            t_stds = [x[1] for x in bucket[d]]
            rows.append((d, statistics.mean(t_means), statistics.mean(t_stds)))
        merged[key] = rows
    return per_cfg, merged


def extrapolate_at_100(points, idx_mean, idx_std):
    # points sorted by distance
    if not points:
        return None, None
    if len(points) == 1:
        return points[0][idx_mean], points[0][idx_std]

    d1, m1, s1 = points[-2][0], points[-2][idx_mean], points[-2][idx_std]
    d2, m2, s2 = points[-1][0], points[-1][idx_mean], points[-1][idx_std]
    if d2 == d1:
        return m2, s2

    # linear extrapolation based on observed difference in last segment
    slope = (m2 - m1) / (d2 - d1)
    m100 = m2 + slope * (TARGET_DISTANCE - d2)
    s100 = (s1 + s2) / 2.0
    return m100, s100


def kalman(prev, val):
    if prev is None:
        return val
    k = 0.5
    return prev + k * (val - prev)


def generate_file(path, sf, bw, tp, rssi_mean, rssi_std, time_mean, time_std):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    random.seed(hash((path, sf, bw, tp)) % (2 ** 32))

    rows = []
    rows.append(["timestamp", "rssi", "kalman_rssi", "sma_rssi", "time_between_messages_ms", "payload"])

    start = datetime.now()
    cfg_rssi = int(math.floor(rssi_mean))
    rows.append(
        [
            start.isoformat(),
            f"{cfg_rssi}.0",
            f"{cfg_rssi:.4f}",
            f"{cfg_rssi}.0",
            "",
            f"CFG sf={sf} sbw={bw} tp={tp} [GENERATED_100M]",
        ]
    )

    cur_ts = start
    k_val = float(cfg_rssi)
    window = [float(cfg_rssi)]
    counter = 1000000

    for _ in range(100):
        rssi = int(math.floor(random.gauss(rssi_mean, max(0.8, rssi_std))))
        k_val = kalman(k_val, float(rssi))
        window.append(float(rssi))
        if len(window) > 4:
            window.pop(0)
        sma = statistics.mean(window)

        t = int(max(100, round(random.gauss(time_mean, max(10.0, time_std)))))
        cur_ts = cur_ts + timedelta(milliseconds=t)
        counter += 1
        payload = f"{counter},100,8," + ",".join([f"{random.randint(0,255):02X}" for _ in range(8)])
        rows.append([cur_ts.isoformat(), f"{rssi}.0", f"{k_val:.4f}", f"{sma:.1f}", str(t), payload])

    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def main():
    per_cfg, per_sfbw = collect_models()
    out_dir = os.path.join(ROOT, "distance_100.0m")
    os.makedirs(out_dir, exist_ok=True)

    generated = 0
    for cfg in expected_configs():
        sf, bw, tp = cfg
        out_path = os.path.join(out_dir, f"SF{sf}_BW{bw}_TP{tp}.csv")
        if os.path.exists(out_path):
            rows = read_rows(out_path)
            if len(rows) == 102:
                continue

        cfg_points = per_cfg.get(cfg, [])
        if cfg_points:
            rssi_mean, rssi_std = extrapolate_at_100(cfg_points, idx_mean=1, idx_std=2)
        else:
            # fallback from TP12 and TP adjustment
            base_points = per_cfg.get((sf, bw, 12), [])
            if base_points:
                base_rssi_mean, base_rssi_std = extrapolate_at_100(base_points, 1, 2)
                rssi_mean = base_rssi_mean + (tp - 12) * 0.6
                rssi_std = base_rssi_std
            else:
                rssi_mean, rssi_std = -95.0, 2.5

        # time model is TP-invariant: use (sf,bw)
        sfbw_points = per_sfbw.get((sf, bw), [])
        if sfbw_points:
            time_mean, time_std = extrapolate_at_100(sfbw_points, idx_mean=1, idx_std=2)
        else:
            base = 1500.0 * ((2 ** sf / bw) / (2 ** 7 / 125000))
            time_mean, time_std = base, max(20.0, base * 0.04)

        generate_file(out_path, sf, bw, tp, rssi_mean, rssi_std, time_mean, time_std)
        generated += 1

    print(f"Generated files at 100m: {generated}")


if __name__ == "__main__":
    main()

