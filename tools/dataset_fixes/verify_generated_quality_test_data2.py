import csv
import os
import re


def is_valid_hex_payload(payload_str):
    if not payload_str:
        return False
    parts = payload_str.strip('"').split(",")
    for part in parts:
        if part and re.match(r"^[0-9A-F]+$", part) is None:
            return False
    return True


def main():
    root = r"C:\Users\ruben\Documents\LoRa Project\test_data2"
    rssi_ok_re = re.compile(r"^-?\d+\.0$")

    generated_files = 0
    bad_rssi_rows = 0
    invalid_payload_rows = 0
    total_generated_rows = 0

    for dn in os.listdir(root):
        folder = os.path.join(root, dn)
        if not (os.path.isdir(folder) and dn.startswith("distance_")):
            continue
        for fn in os.listdir(folder):
            if not fn.endswith(".csv"):
                continue
            fp = os.path.join(folder, fn)
            rows = list(csv.reader(open(fp, "r", encoding="utf-8")))
            if len(rows) <= 1 or len(rows[1]) <= 5:
                continue
            if "[GENERATED_FROM_RAW]" not in rows[1][5]:
                continue

            generated_files += 1
            for row in rows[2:]:
                if len(row) < 6:
                    continue
                total_generated_rows += 1
                if rssi_ok_re.match(row[1] or "") is None:
                    bad_rssi_rows += 1
                if not is_valid_hex_payload(row[5]):
                    invalid_payload_rows += 1

    print(f"generated_files={generated_files}")
    print(f"bad_rssi_rows={bad_rssi_rows}")
    print(f"invalid_payload_rows={invalid_payload_rows}")
    print(f"total_generated_rows={total_generated_rows}")


if __name__ == "__main__":
    main()

