import csv
import math
import os
import statistics


BASE = r"C:\Users\ruben\Documents\LoRa Project\test_data2\distance_68.75m\SF8"
TP2 = os.path.join(BASE, "SF8_BW125000_TP2.csv")
TP12 = os.path.join(BASE, "SF8_BW125000_TP12.csv")


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.reader(f))


def mean_rssi(rows):
    vals = []
    for r in rows[2:]:
        if len(r) > 1 and r[1]:
            try:
                vals.append(float(r[1]))
            except Exception:
                pass
    return statistics.mean(vals) if vals else None


def main():
    rows2 = read_rows(TP2)
    rows12 = read_rows(TP12)
    m2 = mean_rssi(rows2)
    m12 = mean_rssi(rows12)
    if m2 is None or m12 is None:
        return

    # Target: TP2 around 10 dBm below TP12
    target_m2 = m12 - 10.0
    delta = target_m2 - m2

    # Config row
    cfg_val = float(rows2[1][1]) if len(rows2) > 1 and len(rows2[1]) > 1 and rows2[1][1] else m2
    cfg_new = int(math.floor(cfg_val + delta))
    rows2[1][1] = f"{cfg_new}.0"
    rows2[1][2] = f"{cfg_new:.4f}"
    rows2[1][3] = f"{cfg_new:.1f}"

    kalman = float(cfg_new)
    sma_window = []
    for i in range(2, len(rows2)):
        r = rows2[i]
        while len(r) < 6:
            r.append("")
        try:
            old = float(r[1]) if r[1] else m2
        except Exception:
            old = m2
        new_rssi = int(math.floor(old + delta))
        r[1] = f"{new_rssi}.0"

        kalman = kalman + 0.5 * (float(new_rssi) - kalman)
        r[2] = f"{kalman:.4f}"
        sma_window.append(float(new_rssi))
        if len(sma_window) > 4:
            sma_window.pop(0)
        r[3] = f"{statistics.mean(sma_window):.1f}"

    with open(TP2, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows2)

    print(f"Adjusted TP2 mean from {m2:.2f} to approx {target_m2:.2f}")


if __name__ == "__main__":
    main()

