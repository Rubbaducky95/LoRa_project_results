"""
Regenerate airtime_by_sf_bw_payload.csv for raw LoRa (0 overhead).
Uses formula from ifTNT/lora-air-time (https://github.com/ifTNT/lora-air-time)
which is designed for pure LoRa modem, not LoRaWAN.
CR 4/6, preamble 8, CRC OFF, explicit header ON, low_data_rate_opt when T_sym > 16 ms.
"""
import csv
import math
import os

WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
AIRTIME_FILE = os.path.join(WORKSPACE, "airtime_by_sf_bw_payload.csv")

BW_VALUES = [62500, 125000, 250000, 500000]
SF_VALUES = [7, 8, 9, 10, 11, 12]
PAYLOAD_SIZES = [34, 35, 36, 37, 38, 39]


def lora_airtime_ms(
    payload_bytes: int,
    sf: int,
    bw_hz: int,
    *,
    crc: bool = False,
    explicit_header: bool = True,
    preamble_len: int = 8,
    coding_rate: int = 6,
) -> float:
    """
    Raw LoRa airtime in ms. ifTNT/lora-air-time formula.
    coding_rate: 5=4/5, 6=4/6, 7=4/7, 8=4/8.
    low_data_rate_opt: when T_sym > 16 ms.
    """
    bw_khz = bw_hz / 1000
    symbol_time = (2**sf) / bw_khz  # ms

    # ifTNT payload_bit formula
    payload_bit = 8 * payload_bytes - 4 * sf + 8
    payload_bit += 16 if crc else 0
    payload_bit += 20 if explicit_header else 0
    payload_bit = max(payload_bit, 0)

    # low_data_rate_opt: when T_sym > 16 ms (Semtech recommendation)
    low_dr = symbol_time > 16
    bits_per_symbol = sf - 2 if low_dr else sf

    payload_symbol = math.ceil(payload_bit / 4 / bits_per_symbol) * coding_rate
    payload_symbol += 8  # SyncWord overhead

    n_preamble = preamble_len + 4.25
    t_preamble = n_preamble * symbol_time
    t_payload = payload_symbol * symbol_time

    return round(t_preamble + t_payload, 1)


def main():
    rows = [["bw", "payload_size", "sf", "airtime_ms"]]
    for bw in BW_VALUES:
        for pl in PAYLOAD_SIZES:
            for sf in SF_VALUES:
                airtime = lora_airtime_ms(pl, sf, bw)
                rows.append([bw, pl, sf, airtime])

    with open(AIRTIME_FILE, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"Wrote {len(rows) - 1} rows to {AIRTIME_FILE}")
    print("Raw LoRa (ifTNT formula): CR 4/6, preamble 8, CRC OFF, explicit header ON, low_dr when T_sym>16ms")


if __name__ == "__main__":
    main()
