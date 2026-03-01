import csv
import os
from collections import defaultdict


IN_CSV = r"C:\Users\ruben\Documents\LoRa Project\results\energy\packet_energy_by_config.csv"
OUT_CSV = r"C:\Users\ruben\Documents\LoRa Project\results\energy\packet_energy_by_config.csv"
BACKUP_CSV = r"C:\Users\ruben\Documents\LoRa Project\results\energy\packet_energy_by_config_by_distance_backup.csv"


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    if not os.path.exists(IN_CSV):
        raise FileNotFoundError(f"Input file not found: {IN_CSV}")

    rows = list(csv.DictReader(open(IN_CSV, "r", encoding="utf-8")))
    if not rows:
        print("Input file is empty, nothing to do.")
        return

    # Backup current distance-resolved file once.
    if not os.path.exists(BACKUP_CSV):
        with open(BACKUP_CSV, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    # Aggregate by config + packet index (distance removed)
    acc = defaultdict(lambda: {
        "n": 0,
        "sum_time_ms": 0.0,
        "sum_uptime_s": 0.0,
        "sum_energy_j": 0.0,
        "sum_eirp_energy_j": 0.0,
        "statuses": defaultdict(int),
    })

    for r in rows:
        key = (
            int(r["sf"]),
            int(r["bw"]),
            int(r["tp"]),
            int(r["packet_index"]),
        )
        a = acc[key]
        a["n"] += 1
        a["sum_time_ms"] += to_float(r.get("time_between_messages_ms", "")) or 0.0
        a["sum_uptime_s"] += to_float(r.get("tx_uptime_s", "")) or 0.0
        a["sum_energy_j"] += to_float(r.get("packet_energy_j", "")) or 0.0
        a["sum_eirp_energy_j"] += to_float(r.get("packet_eirp_energy_j", "")) or 0.0
        status = (r.get("payload_status") or "").strip() or "UNKNOWN"
        a["statuses"][status] += 1

    out_rows = []
    for (sf, bw, tp, packet_index), a in sorted(acc.items()):
        n = a["n"]
        # Majority payload status across distances
        payload_status = max(a["statuses"].items(), key=lambda kv: kv[1])[0]
        out_rows.append([
            sf,
            bw,
            tp,
            packet_index,
            n,
            f"{a['sum_time_ms'] / n:.3f}",
            f"{a['sum_uptime_s'] / n:.6f}",
            f"{a['sum_energy_j'] / n:.9f}",
            f"{a['sum_eirp_energy_j'] / n:.9f}",
            payload_status,
        ])

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sf",
            "bw",
            "tp",
            "packet_index",
            "num_distances_aggregated",
            "avg_time_between_messages_ms",
            "avg_tx_uptime_s",
            "avg_packet_energy_j",
            "avg_packet_eirp_energy_j",
            "payload_status",
        ])
        w.writerows(out_rows)

    print(f"Wrote distance-agnostic packet energy CSV: {OUT_CSV}")
    print(f"Rows written: {len(out_rows)}")
    print(f"Backup saved at: {BACKUP_CSV}")


if __name__ == "__main__":
    main()

