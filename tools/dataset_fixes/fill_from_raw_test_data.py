import csv
import math
import os
import random
import re
import statistics
from collections import defaultdict
from datetime import datetime, timedelta


RAW_DIR = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
TARGET_DIR = r"C:\Users\ruben\Documents\LoRa Project\test_data"

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
EXPECTED_DATA_ROWS = 100


def calculate_kalman_rssi(prev_kalman, current_rssi, r=1.0):
    if prev_kalman is None:
        return current_rssi
    p = 1.0
    k = p / (p + r)
    return prev_kalman + k * (current_rssi - prev_kalman)


def calculate_sma_rssi(recent_values, window=4):
    if not recent_values:
        return None
    vals = recent_values[-window:]
    return statistics.mean(vals)


def expected_configs():
    return [
        (sf, bw, tp)
        for sf in SF_VALUES
        for bw in BW_VALUES
        for tp in TP_VALUES
        if (sf, bw, tp) not in EXCLUDED
    ]


def parse_distance_from_folder_name(folder_name):
    match = re.match(r"^distance_([\d.]+)m?$", folder_name)
    if not match:
        return None
    return float(match.group(1))


def parse_config_filename(filename):
    match = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    if not match:
        return None
    return tuple(map(int, match.groups()))


def list_distance_folders(base_dir):
    out = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if not (os.path.isdir(full) and name.startswith("distance_")):
            continue
        dist = parse_distance_from_folder_name(name)
        if dist is not None:
            out.append((name, full, dist))
    return sorted(out, key=lambda x: x[2])


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.reader(f))


def extract_numeric_series(rows):
    rssi_values = []
    time_values = []
    for row in rows[2:102]:
        if len(row) < 5:
            continue
        try:
            if row[1]:
                rssi_values.append(float(row[1]))
        except ValueError:
            pass
        try:
            if row[4]:
                time_values.append(float(row[4]))
        except ValueError:
            pass
    return rssi_values, time_values


def build_raw_models():
    # Per full config, collect distance->stats
    cfg_by_distance = defaultdict(list)  # (sf,bw,tp) -> [(distance,rssi_mean,rssi_std,time_mean,time_std), ...]
    # Time baseline by (sf,bw), TP-independent
    time_by_sfbw = defaultdict(list)  # (sf,bw) -> [(distance,time_mean,time_std), ...]

    for _, folder, distance in list_distance_folders(RAW_DIR):
        for filename in os.listdir(folder):
            parsed = parse_config_filename(filename)
            if not parsed:
                continue
            sf, bw, tp = parsed
            if tp not in TP_VALUES:
                continue
            path = os.path.join(folder, filename)
            try:
                rows = read_csv_rows(path)
            except Exception:
                continue
            if len(rows) < 3:
                continue

            rssi_values, time_values = extract_numeric_series(rows)
            if not rssi_values or not time_values:
                continue

            rssi_mean = statistics.mean(rssi_values)
            rssi_std = statistics.stdev(rssi_values) if len(rssi_values) > 1 else 1.5
            time_mean = statistics.mean(time_values)
            time_std = statistics.stdev(time_values) if len(time_values) > 1 else max(20.0, time_mean * 0.03)

            cfg_by_distance[(sf, bw, tp)].append((distance, rssi_mean, rssi_std, time_mean, time_std))
            time_by_sfbw[(sf, bw)].append((distance, time_mean, time_std))

    # Sort each list by distance
    for key in list(cfg_by_distance.keys()):
        cfg_by_distance[key].sort(key=lambda x: x[0])
    for key in list(time_by_sfbw.keys()):
        # average duplicates at same distance across TP
        per_dist = defaultdict(list)
        for d, tmean, tstd in time_by_sfbw[key]:
            per_dist[d].append((tmean, tstd))
        merged = []
        for d in sorted(per_dist.keys()):
            tmeans = [x[0] for x in per_dist[d]]
            tstds = [x[1] for x in per_dist[d]]
            merged.append((d, statistics.mean(tmeans), statistics.mean(tstds)))
        time_by_sfbw[key] = merged

    return cfg_by_distance, time_by_sfbw


def estimate_rssi(distance, sf, bw, tp, cfg_by_distance):
    points = cfg_by_distance.get((sf, bw, tp), [])
    if not points:
        # Fallback from TP=12 then adjust by TP delta
        base_points = cfg_by_distance.get((sf, bw, 12), [])
        if base_points:
            base_rssi, base_std = estimate_rssi(distance, sf, bw, 12, cfg_by_distance)
            return base_rssi + (tp - 12) * 0.6, base_std
        return -95.0, 2.5

    distances = [p[0] for p in points]
    if len(points) == 1:
        d0, r0, s0, _, _ = points[0]
        n = 2.5
        rssi = r0 - 10 * n * (math.log10(distance / d0) if d0 > 0 and distance > 0 else 0.0)
        return rssi, s0

    # Bracket interpolation with path-loss-derived exponent
    if distance <= distances[0]:
        d1, r1, s1, _, _ = points[0]
        d2, r2, _, _, _ = points[1]
    elif distance >= distances[-1]:
        d1, r1, s1, _, _ = points[-2]
        d2, r2, _, _, _ = points[-1]
    else:
        idx = 0
        for i in range(len(points) - 1):
            if points[i][0] <= distance <= points[i + 1][0]:
                idx = i
                break
        d1, r1, s1, _, _ = points[idx]
        d2, r2, _, _, _ = points[idx + 1]

    if d1 > 0 and d2 > 0 and d2 != d1:
        try:
            n = -(r2 - r1) / (10 * math.log10(d2 / d1))
            n = max(2.0, min(4.0, n))
        except Exception:
            n = 2.5
    else:
        n = 2.5
    rssi = r1 - 10 * n * (math.log10(distance / d1) if d1 > 0 and distance > 0 else 0.0)
    return rssi, s1


def estimate_time(distance, sf, bw, time_by_sfbw):
    # User requested behavior: time increases with SF, decreases with BW.
    # Primary source is same (sf,bw) trend from raw data.
    points = time_by_sfbw.get((sf, bw), [])
    if points:
        if len(points) == 1:
            return points[0][1], points[0][2]
        if distance <= points[0][0]:
            return points[0][1], points[0][2]
        if distance >= points[-1][0]:
            return points[-1][1], points[-1][2]
        for i in range(len(points) - 1):
            d1, t1, s1 = points[i]
            d2, t2, s2 = points[i + 1]
            if d1 <= distance <= d2 and d2 != d1:
                r = (distance - d1) / (d2 - d1)
                return t1 + (t2 - t1) * r, (s1 + s2) / 2

    # Fallback: derive from nearest available (sf,bw) using airtime ratio
    candidates = []
    for (s, b), pts in time_by_sfbw.items():
        if not pts:
            continue
        d0, t0, s0 = min(pts, key=lambda x: abs(x[0] - distance))
        candidates.append((abs(s - sf) + abs(b - bw) / 125000.0, s, b, t0, s0))
    if candidates:
        _, s_base, b_base, t_base, s_base_std = min(candidates, key=lambda x: x[0])
        ratio = (2 ** sf / bw) / (2 ** s_base / b_base)
        est = t_base * ratio
        return est, max(20.0, s_base_std * 0.7)

    # Hard fallback
    est = 1500.0 * ((2 ** sf / bw) / (2 ** 7 / 125000))
    return est, max(20.0, est * 0.04)


def generate_payload(counter):
    hex_values = ",".join([f"{random.randint(0, 255):02X}" for _ in range(8)])
    return f"{counter},100,8,{hex_values}"


def generate_file(filepath, distance, sf, bw, tp, cfg_by_distance, time_by_sfbw):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    random.seed(hash((filepath, distance, sf, bw, tp)) % (2 ** 32))

    rssi_mean, rssi_std = estimate_rssi(distance, sf, bw, tp, cfg_by_distance)
    time_mean, time_std = estimate_time(distance, sf, bw, time_by_sfbw)

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
            f"CFG sf={sf} sbw={bw} tp={tp} [GENERATED_FROM_RAW]",
        ]
    )

    last_dt = start
    last_kalman = float(cfg_rssi)
    recent = [float(cfg_rssi)]
    payload_counter = 1000000

    for _ in range(EXPECTED_DATA_ROWS):
        rssi = int(math.floor(random.gauss(rssi_mean, max(0.8, rssi_std))))
        last_kalman = calculate_kalman_rssi(last_kalman, float(rssi))
        recent.append(float(rssi))
        if len(recent) > 4:
            recent.pop(0)
        sma = calculate_sma_rssi(recent)
        if sma is None:
            sma = float(rssi)

        t_ms = int(max(100, round(random.gauss(time_mean, max(10.0, time_std)))))
        last_dt = last_dt + timedelta(milliseconds=t_ms)

        payload_counter += 1
        rows.append(
            [
                last_dt.isoformat(),
                f"{rssi}.0",
                f"{last_kalman:.4f}",
                f"{sma:.1f}",
                str(t_ms),
                generate_payload(payload_counter),
            ]
        )

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def main():
    cfg_by_distance, time_by_sfbw = build_raw_models()
    exp = expected_configs()

    distance_folders = list_distance_folders(TARGET_DIR)
    generated = 0
    skipped = 0

    for folder_name, folder_path, distance in distance_folders:
        for sf, bw, tp in exp:
            filename = f"SF{sf}_BW{bw}_TP{tp}.csv"
            filepath = os.path.join(folder_path, filename)
            if os.path.exists(filepath):
                try:
                    rows = read_csv_rows(filepath)
                    if len(rows) == 102:
                        skipped += 1
                        continue
                except Exception:
                    pass

            generate_file(filepath, distance, sf, bw, tp, cfg_by_distance, time_by_sfbw)
            generated += 1
            print(f"Generated {filepath}")

    print(f"\nDone. Generated: {generated}, skipped complete: {skipped}")


if __name__ == "__main__":
    main()

