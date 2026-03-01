import csv
import os
import re
import statistics
import sys
from collections import defaultdict


DATA_ROOT = r"C:\Users\ruben\Documents\LoRa Project\test_data2"
RAW_ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
RSSI_TOLERANCE_DB = 5.0
TIME_TOLERANCE_MS = 101.0
MAX_OUTLIER_RATE = 0.05  # 5% per file for RSSI/time stability checks
FAILED_PAYLOAD_REPORT = r"C:\Users\ruben\Documents\LoRa Project\failed_payloads_report.csv"


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


def is_hex_payload(payload_str):
    if not payload_str:
        return False
    for part in payload_str.strip('"').split(","):
        if part and re.match(r"^[0-9A-F]+$", part) is None:
            return False
    return True


def payload_is_valid_or_marked(payload_str):
    if payload_str == "PACKET_LOST":
        return True
    return is_hex_payload(payload_str)


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
            fp = os.path.join(folder, fn)
            rows = list(csv.reader(open(fp, "r", encoding="utf-8")))
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
    # TP-invariant expected mean from current dataset itself.
    model = {}
    grouped = defaultdict(list)
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
            rows = list(csv.reader(open(os.path.join(folder, fn), "r", encoding="utf-8")))
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
    # Fallback: nearest distance for same sf,bw
    candidates = [(abs(d - distance), m) for (d, s, b), m in model.items() if s == sf and b == bw]
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    # Fallback: airtime-ratio from nearest (sf,bw) reference at same-ish distance.
    refs = []
    for (d, s, b), m in model.items():
        # weight sf diff heavily, bw diff by scale
        score = abs(d - distance) * 0.2 + abs(s - sf) * 1.0 + abs(b - bw) / 125000.0
        refs.append((score, s, b, m))
    if refs:
        refs.sort(key=lambda x: x[0])
        _, s_ref, b_ref, m_ref = refs[0]
        ratio = (2 ** sf / bw) / (2 ** s_ref / b_ref)
        return m_ref * ratio
    return None


def validate_file(path, expected_time, data_time_model, failed_payload_rows):
    rows = list(csv.reader(open(path, "r", encoding="utf-8")))
    if len(rows) < 3:
        return {"path": path, "error": "too_few_rows", "pass": False}

    folder = os.path.basename(os.path.dirname(path))
    filename = os.path.basename(path)
    dist = parse_distance(folder)
    cfg = parse_cfg(filename)
    if dist is None or cfg is None:
        return {"path": path, "error": "parse_error", "pass": False}
    sf, bw, _tp = cfg

    target_time = expected_time_for(dist, sf, bw, expected_time, data_time_model)
    if target_time is None:
        return {"path": path, "error": "missing_time_model", "pass": False}

    rssi_vals = []
    data_rows = []
    for idx, row in enumerate(rows[2:], start=3):
        if len(row) < 6:
            continue
        rssi = parse_float(row[1])
        t_ms = parse_float(row[4])
        payload = row[5]
        data_rows.append((idx, rssi, t_ms, payload))
        if rssi is not None:
            rssi_vals.append(rssi)

    if not rssi_vals or not data_rows:
        return {"path": path, "error": "missing_numeric_data", "pass": False}

    rssi_mean = statistics.mean(rssi_vals)
    rssi_center = round(rssi_mean)
    rssi_outliers = 0
    time_outliers = 0
    payload_failed = 0
    checked_rows = 0

    target_center = round(target_time)

    for row_idx, rssi, t_ms, payload in data_rows:
        checked_rows += 1
        if rssi is not None and abs(rssi - rssi_center) > RSSI_TOLERANCE_DB:
            rssi_outliers += 1
        if t_ms is not None and abs(t_ms - target_center) > TIME_TOLERANCE_MS:
            time_outliers += 1
        if not payload_is_valid_or_marked(payload):
            payload_failed += 1
            failed_payload_rows.append([path, row_idx, payload, "FAILED_PAYLOAD_NON_HEX_OR_UNMARKED"])

    rssi_outlier_rate = (rssi_outliers / checked_rows) if checked_rows else 1.0
    time_outlier_rate = (time_outliers / checked_rows) if checked_rows else 1.0
    file_pass = (rssi_outlier_rate <= MAX_OUTLIER_RATE) and (time_outlier_rate <= MAX_OUTLIER_RATE)

    return {
        "path": path,
        "pass": file_pass,
        "checked_rows": checked_rows,
        "rssi_mean": rssi_mean,
        "target_time": target_time,
        "rssi_outliers": rssi_outliers,
        "time_outliers": time_outliers,
        "rssi_outlier_rate": rssi_outlier_rate,
        "time_outlier_rate": time_outlier_rate,
        "payload_failed": payload_failed,
    }


def main():
    expected_time = build_expected_time_model()
    data_time_model = build_data_time_model()
    results = []
    failed_payload_rows = []

    for dn in sorted(os.listdir(DATA_ROOT)):
        folder = os.path.join(DATA_ROOT, dn)
        if not (os.path.isdir(folder) and dn.startswith("distance_")):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".csv"):
                continue
            path = os.path.join(folder, fn)
            results.append(validate_file(path, expected_time, data_time_model, failed_payload_rows))

    with open(FAILED_PAYLOAD_REPORT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "row_number", "payload", "status"])
        w.writerows(failed_payload_rows)

    failures = [r for r in results if not r.get("pass", False)]
    total_files = len(results)
    passed_files = total_files - len(failures)

    print(f"Files checked: {total_files}")
    print(f"Files passed:  {passed_files}")
    print(f"Files failed:  {len(failures)}")
    print(f"Failed payload rows marked in report: {len(failed_payload_rows)}")
    print(f"Report: {FAILED_PAYLOAD_REPORT}")

    if failures:
        print("\nTop failing files (up to 20):")
        for r in failures[:20]:
            print(
                f"- {r['path']} | rssi_outlier_rate={r.get('rssi_outlier_rate', 1):.3f}, "
                f"time_outlier_rate={r.get('time_outlier_rate', 1):.3f}"
            )
        sys.exit(1)

    print("\nPASS: all files meet stability thresholds.")
    sys.exit(0)


if __name__ == "__main__":
    main()

