import csv
import os
import re


ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
CSV_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")

DROP_COLUMNS = {"power_consumption_w", "energy_per_packet_kwh"}


def process_csv(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8")))
    if not rows:
        return False
    header = rows[0]
    keep_idx = [i for i, name in enumerate(header) if name not in DROP_COLUMNS]
    if len(keep_idx) == len(header):
        return False

    new_rows = []
    for row in rows:
        # Guard short rows
        padded = row + [""] * (len(header) - len(row))
        new_rows.append([padded[i] for i in keep_idx])

    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(new_rows)
    return True


def main():
    updated = 0
    for root, _, files in os.walk(ROOT):
        for fn in files:
            if not CSV_RE.match(fn):
                continue
            fp = os.path.join(root, fn)
            if process_csv(fp):
                updated += 1
    print(f"Updated curated CSV files: {updated}")


if __name__ == "__main__":
    main()

