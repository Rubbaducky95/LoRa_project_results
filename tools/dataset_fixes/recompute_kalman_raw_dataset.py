import csv
import os
import re


ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
CSV_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")

# 1D Kalman parameters (tunable)
Q = 0.1  # process noise
R = 1.0  # measurement noise
P0 = 1.0  # initial estimate covariance


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None


def ensure_col(header, name):
    if name not in header:
        header.append(name)
    return header.index(name)


def process_file(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8")))
    if len(rows) < 3:
        return False

    header = rows[0]
    if "rssi" not in header:
        return False
    rssi_idx = header.index("rssi")
    kalman_new_idx = ensure_col(header, "kalman_rssi_recalc")

    # Normalize widths
    for i in range(1, len(rows)):
        while len(rows[i]) < len(header):
            rows[i].append("")

    # x0 = average RSSI in this file (data rows)
    z_vals = []
    for r in rows[2:]:
        z = parse_float(r[rssi_idx]) if len(r) > rssi_idx else None
        if z is not None:
            z_vals.append(z)
    if not z_vals:
        return False
    x = sum(z_vals) / len(z_vals)
    p = P0

    # Put x0 on config row for traceability
    rows[1][kalman_new_idx] = f"{x:.4f}"

    # Sequential predict->update per packet row
    for i in range(2, len(rows)):
        r = rows[i]
        z = parse_float(r[rssi_idx]) if len(r) > rssi_idx else None
        if z is None:
            r[kalman_new_idx] = ""
            continue

        # Predict
        x_prior = x
        p_prior = p + Q

        # Update
        k = p_prior / (p_prior + R)
        x = x_prior + k * (z - x_prior)
        p = (1.0 - k) * p_prior

        r[kalman_new_idx] = f"{x:.4f}"

    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    return True


def main():
    updated = 0
    for root, _, files in os.walk(ROOT):
        for fn in files:
            if CSV_RE.match(fn) is None:
                continue
            path = os.path.join(root, fn)
            if process_file(path):
                updated += 1
    print(f"Updated files with kalman_rssi_recalc: {updated}")
    print(f"Params used: Q={Q}, R={R}, P0={P0}")


if __name__ == "__main__":
    main()

