import csv
import os
import shutil
import re


SOURCE_ROOT = r"C:\Users\ruben\Documents\LoRa Project\test_data2"
TARGET_ROOT = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"

VOLTAGE_V = 3.3
CURRENT_A = 0.0308  # 30.8 mA
POWER_W = VOLTAGE_V * CURRENT_A  # 0.10164 W

CSV_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")


def process_csv(path):
    rows = list(csv.reader(open(path, "r", encoding="utf-8")))
    if not rows:
        return False

    # Add column name if needed.
    if "power_consumption_w" not in rows[0]:
        rows[0].append("power_consumption_w")

    # Fill config + packet rows with constant power value.
    for i in range(1, len(rows)):
        row = rows[i]
        # Keep row width aligned to header.
        while len(row) < len(rows[0]) - 1:
            row.append("")
        if len(row) == len(rows[0]) - 1:
            row.append(f"{POWER_W:.5f}")
        else:
            row[-1] = f"{POWER_W:.5f}"

    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    return True


def main():
    if os.path.exists(TARGET_ROOT):
        shutil.rmtree(TARGET_ROOT)
    shutil.copytree(SOURCE_ROOT, TARGET_ROOT)

    updated = 0
    for root, _, files in os.walk(TARGET_ROOT):
        for fn in files:
            if not CSV_RE.match(fn):
                continue
            path = os.path.join(root, fn)
            if process_csv(path):
                updated += 1

    print(f"Created curated dataset: {TARGET_ROOT}")
    print(f"Voltage: {VOLTAGE_V} V, Current: {CURRENT_A} A, Power: {POWER_W:.5f} W")
    print(f"CSV files updated: {updated}")


if __name__ == "__main__":
    main()

