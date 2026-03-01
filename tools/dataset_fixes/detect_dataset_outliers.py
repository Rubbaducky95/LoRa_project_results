import argparse
import csv
import json
import os
import re
import statistics
from collections import Counter


DEFAULT_ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
DEFAULT_OUT_CSV = r"C:\Users\ruben\Documents\LoRa Project\results\qa\dataset_outlier_report.csv"
DEFAULT_OUT_JSON = r"C:\Users\ruben\Documents\LoRa Project\results\qa\dataset_outlier_summary.json"
CSV_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")
HEX_RE = re.compile(r"^[0-9A-F]+$")


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None


def median_abs_deviation(values):
    if not values:
        return 0.0
    med = statistics.median(values)
    dev = [abs(v - med) for v in values]
    return statistics.median(dev)


def modified_z(value, median, mad):
    if mad == 0:
        return 0.0
    return 0.6745 * (value - median) / mad


def payload_valid(payload):
    if payload == "PACKET_LOST":
        return False
    parts = (payload or "").strip('"').split(",")
    for part in parts:
        if part and HEX_RE.match(part) is None:
            return False
    return True


def detect_file_outliers(path, args):
    rows = list(csv.reader(open(path, "r", encoding="utf-8")))
    if len(rows) < 3:
        return [("file_too_short", 0, "len(rows)<3")]

    header = rows[0]
    idx_rssi = header.index("rssi") if "rssi" in header else 1
    idx_t = header.index("time_between_messages_ms") if "time_between_messages_ms" in header else 4
    idx_payload = header.index("payload") if "payload" in header else 5

    rssi_vals = []
    t_vals = []
    parsed_rows = []  # (row_no, rssi, t, payload)
    for i, r in enumerate(rows[2:], start=3):
        rssi = parse_float(r[idx_rssi]) if len(r) > idx_rssi else None
        t = parse_float(r[idx_t]) if len(r) > idx_t else None
        payload = r[idx_payload] if len(r) > idx_payload else ""
        parsed_rows.append((i, rssi, t, payload))
        if rssi is not None:
            rssi_vals.append(rssi)
        if t is not None:
            t_vals.append(t)

    outliers = []

    # RSSI row-level outliers
    if rssi_vals:
        r_med = statistics.median(rssi_vals)
        r_mad = median_abs_deviation(rssi_vals)
        r_mean = statistics.mean(rssi_vals)

        # Flatline check (optional; can be expected when RSSI is quantized/rounded).
        if args.check_rssi_flatline:
            c = Counter(rssi_vals)
            dominant_ratio = max(c.values()) / len(rssi_vals)
            if dominant_ratio >= args.flatline_ratio:
                outliers.append(("rssi_flatline", 0, f"dominant_ratio={dominant_ratio:.3f}"))

        for row_no, rssi, _, _ in parsed_rows:
            if rssi is None:
                continue
            mz = abs(modified_z(rssi, r_med, r_mad))
            dev_abs = abs(rssi - r_mean)
            if mz > args.rssi_mz_thresh or dev_abs > args.rssi_abs_thresh:
                outliers.append(("rssi_outlier", row_no, f"rssi={rssi},mz={mz:.2f},dev={dev_abs:.2f}"))

    # TX interval row-level outliers
    if t_vals:
        t_med = statistics.median(t_vals)
        t_mad = median_abs_deviation(t_vals)
        t_mean = statistics.mean(t_vals)
        for row_no, _, t, _ in parsed_rows:
            if t is None:
                continue
            mz = abs(modified_z(t, t_med, t_mad))
            dev_abs = abs(t - t_mean)
            if mz > args.time_mz_thresh or dev_abs > args.time_abs_thresh:
                outliers.append(("tx_interval_outlier", row_no, f"time_ms={t},mz={mz:.2f},dev={dev_abs:.2f}"))

    # Payload anomalies
    for row_no, _, _, payload in parsed_rows:
        if not payload_valid(payload):
            outliers.append(("payload_invalid", row_no, payload[:120]))

    return outliers


def main():
    parser = argparse.ArgumentParser(description="Scan dataset CSVs for outliers.")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Dataset root folder to scan.")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV, help="Output CSV report path.")
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON, help="Output summary JSON path.")
    parser.add_argument("--rssi-mz-thresh", type=float, default=3.5, help="Modified-Z threshold for RSSI.")
    parser.add_argument("--time-mz-thresh", type=float, default=3.5, help="Modified-Z threshold for TX interval.")
    parser.add_argument("--rssi-abs-thresh", type=float, default=5.0, help="Absolute deviation threshold (dB).")
    parser.add_argument("--time-abs-thresh", type=float, default=100.0, help="Absolute deviation threshold (ms).")
    parser.add_argument("--flatline-ratio", type=float, default=0.95, help="Dominant RSSI ratio for flatline flag.")
    parser.add_argument(
        "--check-rssi-flatline",
        action="store_true",
        help="Enable RSSI flatline detection (off by default).",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    report_rows = []
    file_count = 0
    files_with_outliers = 0
    by_type = Counter()

    for root, _, files in os.walk(args.root):
        for fn in files:
            if CSV_RE.match(fn) is None:
                continue
            file_count += 1
            path = os.path.join(root, fn)
            outliers = detect_file_outliers(path, args)
            if outliers:
                files_with_outliers += 1
            for outlier_type, row_no, details in outliers:
                by_type[outlier_type] += 1
                report_rows.append(
                    [path, fn, outlier_type, row_no, details]
                )

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_path", "file_name", "outlier_type", "row_number", "details"])
        w.writerows(report_rows)

    summary = {
        "root": args.root,
        "files_scanned": file_count,
        "files_with_outliers": files_with_outliers,
        "outlier_count": len(report_rows),
        "outliers_by_type": dict(by_type),
        "report_csv": args.out_csv,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

