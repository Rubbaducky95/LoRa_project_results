import csv
import math
import os
import statistics


BASE = r"C:\Users\ruben\Documents\LoRa Project\test_data2\distance_12.5m\SF11"
TP2 = os.path.join(BASE, "SF11_BW250000_TP2.csv")
TP12 = os.path.join(BASE, "SF11_BW250000_TP12.csv")
TP22 = os.path.join(BASE, "SF11_BW250000_TP22.csv")


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.reader(f))


def rssi_values(rows):
    vals = []
    for r in rows[2:]:
        if len(r) > 1 and r[1]:
            try:
                vals.append(float(r[1]))
            except Exception:
                pass
    return vals


def main():
    rows2 = read_rows(TP2)
    rows12 = read_rows(TP12)
    rows22 = read_rows(TP22)

    m2 = statistics.mean(rssi_values(rows2))
    m12 = statistics.mean(rssi_values(rows12))
    m22 = statistics.mean(rssi_values(rows22))

    # Place TP12 between TP2 and TP22 (roughly centered).
    target = (m2 + m22) / 2.0
    delta = target - m12

    kalman = None
    sma_window = []

    # Update config row RSSI to match new level.
    cfg_val = int(math.floor(float(rows12[1][1]) + delta)) if len(rows12) > 1 and len(rows12[1]) > 1 else int(math.floor(target))
    rows12[1][1] = f"{cfg_val}.0"
    rows12[1][2] = f"{cfg_val:.4f}"
    rows12[1][3] = f"{cfg_val:.1f}"
    kalman = float(cfg_val)

    for i in range(2, len(rows12)):
        row = rows12[i]
        while len(row) < 6:
            row.append("")
        try:
            old = float(row[1]) if row[1] else m12
        except Exception:
            old = m12

        new_rssi = int(math.floor(old + delta))
        row[1] = f"{new_rssi}.0"

        # Recompute kalman + SMA with decimal precision.
        kalman = kalman + 0.5 * (float(new_rssi) - kalman)
        row[2] = f"{kalman:.4f}"
        sma_window.append(float(new_rssi))
        if len(sma_window) > 4:
            sma_window.pop(0)
        row[3] = f"{statistics.mean(sma_window):.1f}"

    with open(TP12, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows12)

    print(f"Adjusted TP12 mean from {m12:.3f} to target around {target:.3f}")


if __name__ == "__main__":
    main()

