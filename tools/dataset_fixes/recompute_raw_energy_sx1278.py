import csv
import os
import re


ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
CSV_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")

# SX1278 typical TX-current model (approximate, module-level guidance).
# TP22 is treated as "max TX setting" proxy in this dataset.
VOLTAGE_V = 3.3
CURRENT_A_BY_TP = {
    2: 0.033,   # 33 mA
    12: 0.060,  # 60 mA (interpolated typical)
    22: 0.120,  # 120 mA (high-power mode proxy)
}


def parse_tp(filename):
    m = CSV_RE.match(filename)
    if not m:
        return None
    return int(m.group(3))


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None


def ensure_col(header, name):
    if name not in header:
        header.append(name)
    return header.index(name)


def main():
    updated = 0
    for root, _, files in os.walk(ROOT):
        for fn in files:
            tp = parse_tp(fn)
            if tp is None or tp not in CURRENT_A_BY_TP:
                continue
            path = os.path.join(root, fn)
            rows = list(csv.reader(open(path, "r", encoding="utf-8")))
            if not rows:
                continue

            header = rows[0]
            j_idx = ensure_col(header, "energy_per_packet_j")

            power_w = VOLTAGE_V * CURRENT_A_BY_TP[tp]

            # Normalize row width
            for i in range(1, len(rows)):
                while len(rows[i]) < len(header):
                    rows[i].append("")

            # config row: no packet interval -> blank
            if len(rows) > 1:
                rows[1][j_idx] = ""

            for i in range(2, len(rows)):
                r = rows[i]
                if len(r) < 5:
                    r[j_idx] = ""
                    continue
                t_ms = parse_float(r[4])
                if t_ms is None:
                    r[j_idx] = ""
                    continue
                e_j = power_w * (t_ms / 1000.0)
                r[j_idx] = f"{e_j:.9f}"

            with open(path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerows(rows)
            updated += 1

    print(f"Updated files: {updated}")
    print("Model: SX1278, V=3.3V, I(TP2/TP12/TP22)=33/60/120 mA")


if __name__ == "__main__":
    main()

