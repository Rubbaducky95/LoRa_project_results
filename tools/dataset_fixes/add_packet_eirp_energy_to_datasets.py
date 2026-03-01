import argparse
import csv
import math
import os


DEFAULT_ROOTS = [
    r"C:\Users\ruben\Documents\LoRa Project\raw_test_data",
    r"C:\Users\ruben\Documents\LoRa Project\dataset",
]


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None


def process_root(root, gain_linear):
    files_updated = 0
    files_skipped = 0
    cells_updated = 0

    for walk_root, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".csv"):
                continue
            path = os.path.join(walk_root, name)
            rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
            if not rows:
                files_skipped += 1
                continue

            header = rows[0]
            if "energy_per_packet_j" not in header:
                files_skipped += 1
                continue

            energy_idx = header.index("energy_per_packet_j")
            if "packet_eirp_energy_j" not in header:
                header.append("packet_eirp_energy_j")
            eirp_idx = header.index("packet_eirp_energy_j")

            changed = 0
            for i in range(1, len(rows)):
                row = rows[i]
                while len(row) < len(header):
                    row.append("")
                e = parse_float(row[energy_idx]) if len(row) > energy_idx else None
                if e is None:
                    row[eirp_idx] = ""
                    continue
                eirp = e * gain_linear
                new_value = f"{eirp:.9f}"
                if row[eirp_idx] != new_value:
                    row[eirp_idx] = new_value
                    changed += 1

            if changed:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    csv.writer(f).writerows(rows)
                files_updated += 1
                cells_updated += changed
            else:
                files_skipped += 1

    return files_updated, files_skipped, cells_updated


def main():
    parser = argparse.ArgumentParser(
        description="Add packet_eirp_energy_j to CSV files from energy_per_packet_j."
    )
    parser.add_argument(
        "--antenna-gain-dbi",
        type=float,
        default=3.0,
        help="Antenna gain in dBi (default: 3.0).",
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        default=DEFAULT_ROOTS,
        help="Dataset roots to process (defaults: raw_test_data and dataset).",
    )
    args = parser.parse_args()

    gain_linear = math.pow(10.0, args.antenna_gain_dbi / 10.0)
    print(f"Using antenna_gain_dbi={args.antenna_gain_dbi:.3f}, gain_linear={gain_linear:.6f}")

    total_updated = 0
    total_skipped = 0
    total_cells = 0
    for root in args.roots:
        if not os.path.isdir(root):
            print(f"Skip missing root: {root}")
            continue
        u, s, c = process_root(root, gain_linear)
        total_updated += u
        total_skipped += s
        total_cells += c
        print(f"[{root}] files_updated={u}, files_skipped={s}, cells_updated={c}")

    print(f"TOTAL files_updated={total_updated}, files_skipped={total_skipped}, cells_updated={total_cells}")


if __name__ == "__main__":
    main()
