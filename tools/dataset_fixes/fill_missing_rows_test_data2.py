import csv
import math
import os
import random
import re
import statistics
from datetime import datetime, timedelta


ROOT = r"C:\Users\ruben\Documents\LoRa Project\test_data2"
TARGET_TOTAL_ROWS = 102


def parse_ts(text):
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def is_hex_payload(payload_str):
    if not payload_str:
        return False
    for part in payload_str.strip('"').split(","):
        if part and re.match(r"^[0-9A-F]+$", part) is None:
            return False
    return True


def generate_valid_payload(counter):
    hex_values = ",".join([f"{random.randint(0, 255):02X}" for _ in range(8)])
    return f"{counter},100,8,{hex_values}"


def generate_invalid_payload(base_payload):
    parts = base_payload.split(",")
    if not parts:
        parts = ["1000000", "100", "8", "AA", "BB", "CC", "DD", "EE"]
    idx = random.randint(0, len(parts) - 1)
    bad_char = random.choice(["<", ">", "$", "g", "p", "?", "&"])
    part = parts[idx] if parts[idx] else "0"
    pos = random.randint(0, len(part))
    if pos >= len(part):
        parts[idx] = part + bad_char
    else:
        parts[idx] = part[:pos] + bad_char + part[pos + 1 :]
    return ",".join(parts)


def fill_file(filepath):
    rows = list(csv.reader(open(filepath, "r", encoding="utf-8")))
    if len(rows) >= TARGET_TOTAL_ROWS or len(rows) < 2:
        return False

    data_rows = rows[2:]
    rssi_vals = []
    time_vals = []
    invalid_count = 0
    payload_counter = 1000000
    last_ts = parse_ts(rows[1][0] if len(rows[1]) > 0 else "")

    for row in data_rows:
        if len(row) < 6:
            continue
        try:
            rssi_vals.append(float(row[1]))
        except Exception:
            pass
        try:
            if row[4]:
                time_vals.append(float(row[4]))
        except Exception:
            pass
        if not is_hex_payload(row[5]):
            invalid_count += 1
        try:
            payload_counter = max(payload_counter, int((row[5] or "").strip('"').split(",")[0]))
        except Exception:
            pass
        ts = parse_ts(row[0] if len(row) > 0 else "")
        if ts is not None:
            last_ts = ts

    # Use file-local stats as requested.
    rssi_mean = statistics.mean(rssi_vals) if rssi_vals else -90.0
    rssi_std = statistics.stdev(rssi_vals) if len(rssi_vals) > 1 else 1.5
    time_mean = statistics.mean(time_vals) if time_vals else 1500.0
    time_std = statistics.stdev(time_vals) if len(time_vals) > 1 else max(15.0, time_mean * 0.03)
    invalid_ratio = (invalid_count / len(data_rows)) if data_rows else 0.0

    if last_ts is None:
        last_ts = datetime.now()

    random.seed(hash(filepath) % (2 ** 32))
    to_add = TARGET_TOTAL_ROWS - len(rows)

    for _ in range(to_add):
        rssi = int(math.floor(random.gauss(rssi_mean, max(0.8, rssi_std))))
        time_ms = int(max(100, round(random.gauss(time_mean, max(10.0, time_std)))))
        last_ts = last_ts + timedelta(milliseconds=time_ms)

        payload_counter += 1
        payload = generate_valid_payload(payload_counter)
        if random.random() < invalid_ratio:
            payload = generate_invalid_payload(payload)

        rows.append(
            [
                last_ts.isoformat(),
                f"{rssi}.0",
                f"{rssi:.4f}",
                f"{rssi:.1f}",
                str(time_ms),
                payload,
            ]
        )

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    return True


def main():
    updated = 0
    for dn in os.listdir(ROOT):
        folder = os.path.join(ROOT, dn)
        if not (os.path.isdir(folder) and dn.startswith("distance_")):
            continue
        for fn in os.listdir(folder):
            if not fn.endswith(".csv"):
                continue
            fp = os.path.join(folder, fn)
            if fill_file(fp):
                updated += 1
                print(f"Filled: {fp}")
    print(f"Updated files: {updated}")


if __name__ == "__main__":
    main()

