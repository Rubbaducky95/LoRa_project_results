import csv
import os
import re
from collections import defaultdict


ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
SUMMARY_OUT = r"C:\Users\ruben\Documents\LoRa Project\results\energy\raw_energy_summary.csv"
CSV_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")
DEFAULT_POWER_W = 0.10164


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def parse_cfg(filename):
    m = CSV_RE.match(filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None


def ensure_column(header, name):
    if name not in header:
        header.append(name)
    return header.index(name)


def main():
    os.makedirs(os.path.dirname(SUMMARY_OUT), exist_ok=True)
    summary = defaultdict(lambda: {"count": 0, "sum_j": 0.0, "sum_kwh": 0.0})
    updated_files = 0

    for dn in sorted(os.listdir(ROOT)):
        dpath = os.path.join(ROOT, dn)
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
                path = os.path.join(root, fn)
                rows = list(csv.reader(open(path, "r", encoding="utf-8")))
                if not rows:
                    continue

                header = rows[0]
                power_idx = ensure_column(header, "power_consumption_w")
                j_idx = ensure_column(header, "energy_per_packet_j")
                kwh_idx = ensure_column(header, "energy_per_packet_kwh")

                # Normalize row widths to header
                for i in range(1, len(rows)):
                    while len(rows[i]) < len(header):
                        rows[i].append("")

                # Config row (row 1): no packet interval => leave blank energy
                if len(rows) > 1:
                    power_v = parse_float(rows[1][power_idx])
                    if power_v is None:
                        rows[1][power_idx] = f"{DEFAULT_POWER_W:.5f}"
                    rows[1][j_idx] = ""
                    rows[1][kwh_idx] = ""

                key = (distance, sf, bw, tp)

                for i in range(2, len(rows)):
                    r = rows[i]
                    power_w = parse_float(r[power_idx])
                    if power_w is None:
                        power_w = DEFAULT_POWER_W
                        r[power_idx] = f"{power_w:.5f}"

                    t_ms = parse_float(r[4]) if len(r) > 4 else None
                    if t_ms is None:
                        r[j_idx] = ""
                        r[kwh_idx] = ""
                        continue

                    dt_s = t_ms / 1000.0
                    e_j = power_w * dt_s
                    e_kwh = e_j / 3_600_000.0
                    r[j_idx] = f"{e_j:.9f}"
                    r[kwh_idx] = f"{e_kwh:.12f}"

                    summary[key]["count"] += 1
                    summary[key]["sum_j"] += e_j
                    summary[key]["sum_kwh"] += e_kwh

                with open(path, "w", encoding="utf-8", newline="") as f:
                    csv.writer(f).writerows(rows)
                updated_files += 1

    with open(SUMMARY_OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "distance_m",
                "sf",
                "bw",
                "tp",
                "packet_count",
                "total_energy_j",
                "total_energy_kwh",
                "avg_energy_per_packet_j",
                "avg_energy_per_packet_kwh",
            ]
        )
        for (distance, sf, bw, tp), acc in sorted(summary.items()):
            n = acc["count"]
            if n == 0:
                continue
            w.writerow(
                [
                    distance,
                    sf,
                    bw,
                    tp,
                    n,
                    f"{acc['sum_j']:.9f}",
                    f"{acc['sum_kwh']:.12f}",
                    f"{(acc['sum_j'] / n):.9f}",
                    f"{(acc['sum_kwh'] / n):.12f}",
                ]
            )

    print(f"Updated curated CSV files: {updated_files}")
    print(f"Wrote summary: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()

