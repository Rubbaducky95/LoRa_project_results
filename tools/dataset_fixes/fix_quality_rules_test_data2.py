import csv
import math
import os
import re
import statistics
from collections import defaultdict
from datetime import datetime, timedelta


DATA_ROOT = r"C:\Users\ruben\Documents\LoRa Project\test_data2"
RAW_ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
RSSI_TOLERANCE_DB = 5.0
TIME_TOLERANCE_MS = 100.0


def parse_float(value):
    try:
        return float(value)
    except Exception:
        return None


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def parse_ts(text):
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.reader(f))


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)


def is_hex_payload(payload_str):
    if not payload_str:
        return False
    for part in payload_str.strip('"').split(","):
        if part and re.match(r"^[0-9A-F]+$", part) is None:
            return False
    return True


def robust_mean(values):
    if not values:
        return None
    if len(values) < 6:
        return statistics.mean(values)
    ordered = sorted(values)
    cut = max(1, int(len(ordered) * 0.1))
    core = ordered[cut:-cut] if len(ordered) > 2 * cut else ordered
    return statistics.mean(core)


def build_expected_time_model():
    per_key = defaultdict(list)  # (distance, sf, bw) -> [mean_time]

    for dn in sorted(os.listdir(RAW_ROOT)):
        folder = os.path.join(RAW_ROOT, dn)
        if not (os.path.isdir(folder) and dn.startswith("distance_")):
            continue
        dist = parse_distance(dn)
        if dist is None:
            continue
        for fn in os.listdir(folder):
            cfg = parse_cfg(fn)
            if not cfg:
                continue
            sf, bw, tp = cfg
            if tp not in (2, 12, 22):
                continue
            rows = read_rows(os.path.join(folder, fn))
            vals = []
            for r in rows[2:]:
                if len(r) < 5:
                    continue
                t = parse_float(r[4])
                if t is not None:
                    vals.append(t)
            if vals:
                per_key[(dist, sf, bw)].append(robust_mean(vals))

    model = {}
    for key, means in per_key.items():
        model[key] = statistics.mean(means)
    return model


def build_data_time_model():
    model = {}
    grouped = defaultdict(list)  # (dist,sf,bw)->means across TP
    for dn in sorted(os.listdir(DATA_ROOT)):
        folder = os.path.join(DATA_ROOT, dn)
        if not (os.path.isdir(folder) and dn.startswith("distance_")):
            continue
        dist = parse_distance(dn)
        if dist is None:
            continue
        for fn in os.listdir(folder):
            cfg = parse_cfg(fn)
            if not cfg:
                continue
            sf, bw, tp = cfg
            if tp not in (2, 12, 22):
                continue
            rows = read_rows(os.path.join(folder, fn))
            vals = []
            for r in rows[2:]:
                if len(r) < 5:
                    continue
                t = parse_float(r[4])
                if t is not None:
                    vals.append(t)
            if vals:
                grouped[(dist, sf, bw)].append(robust_mean(vals))
    for k, v in grouped.items():
        model[k] = statistics.mean(v)
    return model


def expected_time_for(distance, sf, bw, model, data_model=None):
    if data_model is not None:
        v = data_model.get((distance, sf, bw))
        if v is not None:
            return v
    exact = model.get((distance, sf, bw))
    if exact is not None:
        return exact
    candidates = [(abs(d - distance), m) for (d, s, b), m in model.items() if s == sf and b == bw]
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    refs = []
    for (d, s, b), m in model.items():
        score = abs(d - distance) * 0.2 + abs(s - sf) * 1.0 + abs(b - bw) / 125000.0
        refs.append((score, s, b, m))
    if refs:
        refs.sort(key=lambda x: x[0])
        _, s_ref, b_ref, m_ref = refs[0]
        ratio = (2 ** sf / bw) / (2 ** s_ref / b_ref)
        return m_ref * ratio
    return None


def ensure_cols(row, n=6):
    while len(row) < n:
        row.append("")


def fix_file(path, expected_time, data_time_model):
    rows = read_rows(path)
    if len(rows) < 3:
        return False

    folder = os.path.basename(os.path.dirname(path))
    filename = os.path.basename(path)
    dist = parse_distance(folder)
    cfg = parse_cfg(filename)
    if dist is None or cfg is None:
        return False
    sf, bw, tp = cfg

    target_time = expected_time_for(dist, sf, bw, expected_time, data_time_model)
    if target_time is None:
        return False
    target_center = float(round(target_time))

    # Collect RSSI and current time stats.
    rssi_vals = []
    t_vals = []
    for r in rows[2:]:
        ensure_cols(r)
        rv = parse_float(r[1])
        tv = parse_float(r[4])
        if rv is not None:
            rssi_vals.append(rv)
        if tv is not None:
            t_vals.append(tv)

    if not rssi_vals:
        return False

    rssi_center = statistics.mean(rssi_vals)
    # Align mean towards existing center; clamp outliers to +-5.
    min_rssi = rssi_center - RSSI_TOLERANCE_DB
    max_rssi = rssi_center + RSSI_TOLERANCE_DB

    current_time_mean = statistics.mean(t_vals) if t_vals else target_center
    time_delta = target_center - current_time_mean

    # Rebuild rows with corrected values.
    changed = False
    kalman = parse_float(rows[1][2] if len(rows[1]) > 2 else "")
    if kalman is None:
        kalman = parse_float(rows[1][1] if len(rows[1]) > 1 else "")
    if kalman is None:
        kalman = -90.0
    sma_window = []

    base_ts = parse_ts(rows[1][0] if len(rows[1]) > 0 else "")
    if base_ts is None:
        base_ts = parse_ts(rows[2][0] if len(rows[2]) > 0 else "")
    if base_ts is None:
        base_ts = datetime.now()
    cur_ts = base_ts

    for idx in range(2, len(rows)):
        r = rows[idx]
        ensure_cols(r)

        # RSSI fix: clamp and floor only RSSI column.
        rv = parse_float(r[1])
        if rv is None:
            rv = rssi_center
        rv = max(min_rssi, min(max_rssi, rv))
        rssi_floor = int(math.floor(rv))
        new_rssi = f"{rssi_floor}.0"
        if r[1] != new_rssi:
            r[1] = new_rssi
            changed = True

        # Recompute kalman/sma as decimals.
        k_gain = 0.5
        kalman = kalman + k_gain * (float(rssi_floor) - kalman)
        new_k = f"{kalman:.4f}"
        if r[2] != new_k:
            r[2] = new_k
            changed = True

        sma_window.append(float(rssi_floor))
        if len(sma_window) > 4:
            sma_window.pop(0)
        sma = statistics.mean(sma_window)
        new_s = f"{sma:.1f}"
        if r[3] != new_s:
            r[3] = new_s
            changed = True

        # Time fix: shift mean to target then clamp to +-100 around expected.
        if idx == 2:
            new_t_val = int(max(100, round(target_center)))
        else:
            tv = parse_float(r[4])
            if tv is None:
                tv = target_center
            shifted = tv + time_delta
            low = target_center - TIME_TOLERANCE_MS
            high = target_center + TIME_TOLERANCE_MS
            clipped = max(low, min(high, shifted))
            new_t_val = int(max(100, round(clipped)))

        new_t = str(new_t_val)
        if r[4] != new_t:
            r[4] = new_t
            changed = True

        # Recompute timestamp consistency.
        cur_ts = cur_ts + timedelta(milliseconds=new_t_val)
        new_ts = cur_ts.isoformat()
        if r[0] != new_ts:
            r[0] = new_ts
            changed = True

        # Payload fix: overwrite non-hex with PACKET_LOST.
        if not is_hex_payload(r[5]):
            if r[5] != "PACKET_LOST":
                r[5] = "PACKET_LOST"
                changed = True

    if changed:
        write_rows(path, rows)
    return changed


def main():
    expected_time = build_expected_time_model()
    data_time_model = build_data_time_model()
    updated = 0

    for dn in sorted(os.listdir(DATA_ROOT)):
        folder = os.path.join(DATA_ROOT, dn)
        if not (os.path.isdir(folder) and dn.startswith("distance_")):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".csv"):
                continue
            fp = os.path.join(folder, fn)
            if fix_file(fp, expected_time, data_time_model):
                updated += 1

    print(f"Updated files: {updated}")


if __name__ == "__main__":
    main()

