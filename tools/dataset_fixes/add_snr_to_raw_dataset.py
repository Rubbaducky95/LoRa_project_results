"""
Add SNR column to raw_test_data CSV files.
SNR (dB) = RSSI - noise_floor = RSSI + 174 - 10*log10(BW_Hz)
Noise floor from thermal: -174 + 10*log10(BW_Hz) dBm
"""
import argparse
import csv
import math
import os
import re


WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
RAW_ROOT = os.path.join(WORKSPACE, "raw_test_data")
DATASET_ROOT = os.path.join(WORKSPACE, "dataset")
CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")


def snr_from_rssi_bw(rssi_dbm, bw_hz):
    """SNR (dB) = RSSI - thermal_noise_floor."""
    noise_floor = -174 + 10 * math.log10(bw_hz)
    return rssi_dbm - noise_floor


def parse_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Add SNR column to raw dataset.")
    parser.add_argument("--data-root", default=RAW_ROOT, help="Raw or dataset root.")
    args = parser.parse_args()

    updated = 0
    for dn in sorted(os.listdir(args.data_root)):
        dpath = os.path.join(args.data_root, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        for root, _, files in os.walk(dpath):
            for fn in files:
                m = CFG_RE.match(fn)
                if not m:
                    continue
                sf, bw, tp = int(m.group(1)), int(m.group(2)), int(m.group(3))
                path = os.path.join(root, fn)
                rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
                if not rows:
                    continue
                header = rows[0]
                if "rssi" not in header:
                    continue
                rssi_idx = header.index("rssi")
                if "snr_db" in header:
                    snr_idx = header.index("snr_db")
                else:
                    header.append("snr_db")
                    snr_idx = len(header) - 1
                for i in range(1, len(rows)):
                    while len(rows[i]) < len(header):
                        rows[i].append("")
                    rssi = parse_float(rows[i][rssi_idx])
                    if rssi is not None:
                        snr = snr_from_rssi_bw(rssi, bw)
                        rows[i][snr_idx] = f"{snr:.2f}"
                    else:
                        rows[i][snr_idx] = ""
                with open(path, "w", encoding="utf-8", newline="") as f:
                    csv.writer(f).writerows(rows)
                updated += 1
    print(f"Updated {updated} files with SNR column")


if __name__ == "__main__":
    main()
