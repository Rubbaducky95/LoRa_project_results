import csv
import os
import re

import fill_from_raw_test_data as gen


ROOT = r"C:\Users\ruben\Documents\LoRa Project\test_data2"


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def parse_cfg(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def main():
    gen.RAW_DIR = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
    cfg_by_distance, time_by_sfbw = gen.build_raw_models()

    fixed_trim = 0
    fixed_regen = 0

    for dn in os.listdir(ROOT):
        folder = os.path.join(ROOT, dn)
        if not (os.path.isdir(folder) and dn.startswith("distance_")):
            continue
        distance = parse_distance(dn)
        if distance is None:
            continue

        for fn in os.listdir(folder):
            if not fn.endswith(".csv"):
                continue
            fp = os.path.join(folder, fn)
            rows = list(csv.reader(open(fp, "r", encoding="utf-8")))

            if len(rows) == 102:
                continue
            if len(rows) > 102:
                rows = rows[:102]
                with open(fp, "w", encoding="utf-8", newline="") as f:
                    csv.writer(f).writerows(rows)
                fixed_trim += 1
                continue

            cfg = parse_cfg(fn)
            if cfg is None:
                continue
            sf, bw, tp = cfg
            gen.generate_file(fp, distance, sf, bw, tp, cfg_by_distance, time_by_sfbw)
            fixed_regen += 1

    print(f"Trimmed files (>102): {fixed_trim}")
    print(f"Regenerated files (<102): {fixed_regen}")


if __name__ == "__main__":
    main()

