import argparse
import csv
import os
import re
from collections import defaultdict


DATA_ROOT = r"C:\Users\ruben\Documents\LoRa Project\test_data2"
OUTPUT_PACKET_CSV = r"C:\Users\ruben\Documents\LoRa Project\results\energy\packet_energy_by_config.csv"
OUTPUT_CONFIG_CSV = r"C:\Users\ruben\Documents\LoRa Project\results\energy\config_energy_summary.csv"


DIST_RE = re.compile(r"^distance_([\d.]+)m?$")
FILE_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")
HEX_RE = re.compile(r"^[0-9A-F]+$")


def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_distance(folder_name):
    m = DIST_RE.match(folder_name)
    return float(m.group(1)) if m else None


def parse_cfg(filename):
    m = FILE_RE.match(filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def payload_is_valid(payload):
    if payload == "PACKET_LOST":
        return False
    parts = (payload or "").strip('"').split(",")
    for part in parts:
        if not part:
            continue
        if HEX_RE.match(part) is None:
            return False
    return True


def load_uptime_overrides(path):
    # Optional CSV columns: distance_m,sf,bw,tp,uptime_s
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Uptime override file not found: {path}")
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (
                    float(row["distance_m"]),
                    int(row["sf"]),
                    int(row["bw"]),
                    int(row["tp"]),
                )
                out[key] = float(row["uptime_s"])
            except Exception:
                continue
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compute packet energy per configuration from test_data2."
    )
    parser.add_argument("--antenna-gain-dbi", type=float, required=True, help="Antenna gain in dBi.")
    # Power-input mode (direct watts)
    parser.add_argument("--power-tp2-w", type=float, default=None, help="Sender power consumption at TP=2 (W).")
    parser.add_argument("--power-tp12-w", type=float, default=None, help="Sender power consumption at TP=12 (W).")
    parser.add_argument("--power-tp22-w", type=float, default=None, help="Sender power consumption at TP=22 (W).")
    # Current-input mode (actual electrical consumption): P = V * I
    parser.add_argument("--voltage-v", type=float, default=None, help="Supply voltage in volts (e.g. 3.3).")
    parser.add_argument("--esp32-current-ma", type=float, default=0.0, help="ESP32 current in mA to add to each TP.")
    parser.add_argument("--lora-current-tp2-ma", type=float, default=None, help="LoRa TX current at TP=2 in mA.")
    parser.add_argument("--lora-current-tp12-ma", type=float, default=None, help="LoRa TX current at TP=12 in mA.")
    parser.add_argument("--lora-current-tp22-ma", type=float, default=None, help="LoRa TX current at TP=22 in mA.")
    parser.add_argument(
        "--uptime-overrides-csv",
        type=str,
        default="",
        help="Optional per-config uptime overrides CSV (distance_m,sf,bw,tp,uptime_s).",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(OUTPUT_PACKET_CSV), exist_ok=True)
    # Resolve power model:
    # 1) direct power if provided for all TP values
    # 2) otherwise use current model if voltage + LoRa currents are provided
    direct_power_ok = all(v is not None for v in (args.power_tp2_w, args.power_tp12_w, args.power_tp22_w))
    current_mode_ok = (
        args.voltage_v is not None
        and args.lora_current_tp2_ma is not None
        and args.lora_current_tp12_ma is not None
        and args.lora_current_tp22_ma is not None
    )

    if direct_power_ok:
        power_by_tp = {2: args.power_tp2_w, 12: args.power_tp12_w, 22: args.power_tp22_w}
        current_by_tp_a = {2: None, 12: None, 22: None}
        voltage_v = None
    elif current_mode_ok:
        voltage_v = args.voltage_v
        current_by_tp_a = {
            2: (args.esp32_current_ma + args.lora_current_tp2_ma) / 1000.0,
            12: (args.esp32_current_ma + args.lora_current_tp12_ma) / 1000.0,
            22: (args.esp32_current_ma + args.lora_current_tp22_ma) / 1000.0,
        }
        power_by_tp = {tp: voltage_v * current_by_tp_a[tp] for tp in (2, 12, 22)}
    else:
        raise ValueError(
            "Provide either (--power-tp2-w/--power-tp12-w/--power-tp22-w) OR "
            "(--voltage-v with --lora-current-tp2-ma/--lora-current-tp12-ma/--lora-current-tp22-ma)."
        )
    gain_linear = 10 ** (args.antenna_gain_dbi / 10.0)
    uptime_overrides = load_uptime_overrides(args.uptime_overrides_csv)

    packet_rows = []
    summary_acc = defaultdict(lambda: {"count": 0, "sum_energy_j": 0.0, "sum_eirp_energy_j": 0.0})

    for dn in sorted(os.listdir(DATA_ROOT)):
        dpath = os.path.join(DATA_ROOT, dn)
        if not (os.path.isdir(dpath) and dn.startswith("distance_")):
            continue
        distance = parse_distance(dn)
        if distance is None:
            continue

        for root, _, files in os.walk(dpath):
            for fn in files:
                cfg = parse_cfg(fn)
                if cfg is None:
                    continue
                sf, bw, tp = cfg
                if tp not in power_by_tp:
                    continue

                path = os.path.join(root, fn)
                rows = list(csv.reader(open(path, "r", encoding="utf-8")))
                if len(rows) < 3:
                    continue

                cfg_key = (distance, sf, bw, tp)
                power_w = power_by_tp[tp]
                uptime_override = uptime_overrides.get(cfg_key)

                packet_idx = 0
                for r in rows[2:]:
                    if len(r) < 6:
                        continue
                    packet_idx += 1
                    t_ms = parse_float(r[4])
                    if uptime_override is not None:
                        tx_uptime_s = uptime_override
                    else:
                        tx_uptime_s = (t_ms / 1000.0) if t_ms is not None else None
                    if tx_uptime_s is None:
                        continue

                    energy_j = power_w * tx_uptime_s
                    eirp_energy_j = energy_j * gain_linear
                    payload_status = "OK" if payload_is_valid(r[5]) else "PACKET_LOST"

                    packet_rows.append(
                        [
                            distance,
                            sf,
                            bw,
                            tp,
                            packet_idx,
                            r[0],
                            r[4],
                            f"{tx_uptime_s:.6f}",
                            "" if voltage_v is None else f"{voltage_v:.6f}",
                            "" if current_by_tp_a[tp] is None else f"{current_by_tp_a[tp]:.6f}",
                            f"{power_w:.6f}",
                            f"{args.antenna_gain_dbi:.3f}",
                            f"{energy_j:.9f}",
                            f"{eirp_energy_j:.9f}",
                            payload_status,
                        ]
                    )

                    acc = summary_acc[cfg_key]
                    acc["count"] += 1
                    acc["sum_energy_j"] += energy_j
                    acc["sum_eirp_energy_j"] += eirp_energy_j

    with open(OUTPUT_PACKET_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "distance_m",
                "sf",
                "bw",
                "tp",
                "packet_index",
                "timestamp",
                "time_between_messages_ms",
                "tx_uptime_s",
                "voltage_v",
                "sender_current_a",
                "sender_power_w",
                "antenna_gain_dbi",
                "packet_energy_j",
                "packet_eirp_energy_j",
                "payload_status",
            ]
        )
        w.writerows(packet_rows)

    with open(OUTPUT_CONFIG_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "distance_m",
                "sf",
                "bw",
                "tp",
                "packet_count",
                "avg_packet_energy_j",
                "avg_packet_eirp_energy_j",
            ]
        )
        for (distance, sf, bw, tp), acc in sorted(summary_acc.items()):
            n = acc["count"]
            if n == 0:
                continue
            w.writerow(
                [
                    distance,
                    sf,
                    bw,
                    tp,
                    n,
                    f"{(acc['sum_energy_j'] / n):.9f}",
                    f"{(acc['sum_eirp_energy_j'] / n):.9f}",
                ]
            )

    print(f"Wrote packet-level CSV: {OUTPUT_PACKET_CSV}")
    print(f"Wrote config summary CSV: {OUTPUT_CONFIG_CSV}")
    print(f"Total packet rows: {len(packet_rows)}")


if __name__ == "__main__":
    main()

