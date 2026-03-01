import csv
import os
import re
from typing import Dict, Optional, Tuple


WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
CURRENT_FILE = os.path.join(WORKSPACE, "currents_tx_power.csv")
AIRTIME_FILE = os.path.join(WORKSPACE, "airtime_by_sf_bw_payload.csv")
DATA_ROOTS = [
    os.path.join(WORKSPACE, "dataset"),
    os.path.join(WORKSPACE, "raw_test_data"),
]

CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")


def parse_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def parse_int(v: str) -> Optional[int]:
    try:
        return int(float(v))
    except Exception:
        return None


def ensure_col(header, name: str) -> int:
    if name not in header:
        header.append(name)
    return header.index(name)


def load_currents(path: str) -> Dict[int, Tuple[float, float, float]]:
    # tx_power -> (voltage_v, min_current_a, max_current_a)
    out: Dict[int, Tuple[float, float, float]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tp = parse_int(row.get("tx_power", ""))
            v = parse_float(row.get("voltage", ""))
            i_min = parse_float(row.get("min_current", ""))
            i_max = parse_float(row.get("max_current", ""))
            if tp is None or v is None or i_min is None or i_max is None:
                continue
            out[tp] = (v, i_min, i_max)
    return out


def load_airtime(path: str) -> Dict[Tuple[int, int, int], float]:
    # (bw, payload_size, sf) -> airtime_ms
    out: Dict[Tuple[int, int, int], float] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # Header currently lower-case, but keep robust matching.
        for row in reader:
            bw = parse_int(row.get("bw", row.get("BW", "")))
            payload_size = parse_int(row.get("payload_size", row.get("payload_size_bytes", "")))
            sf = parse_int(row.get("sf", row.get("SF", "")))
            airtime_ms = parse_float(row.get("airtime_ms", ""))
            if bw is None or payload_size is None or sf is None or airtime_ms is None:
                continue
            out[(bw, payload_size, sf)] = airtime_ms
    return out


def get_airtime_ms(airtimes: Dict[Tuple[int, int, int], float], bw: int, payload_size: int, sf: int) -> Optional[float]:
    """Lookup airtime; if exact payload not in table, use closest available (36, 37, 38)."""
    key = (bw, payload_size, sf)
    if key in airtimes:
        return airtimes[key]
    # Fallback: try closest payload sizes (table has 34-39)
    candidates = sorted([34, 35, 36, 37, 38, 39], key=lambda p: abs(p - payload_size))
    for fallback_pl in candidates:
        alt = airtimes.get((bw, fallback_pl, sf))
        if alt is not None:
            return alt
    return None


def cfg_from_filename(filename: str) -> Optional[Tuple[int, int, int]]:
    m = CFG_RE.match(filename)
    if not m:
        return None
    sf, bw, tp = map(int, m.groups())
    return sf, bw, tp


def update_file(
    path: str,
    sf: int,
    bw: int,
    tp: int,
    currents,
    airtimes,
    initial_last_valid_payload_size: Optional[int] = None,
) -> Tuple[int, int, Optional[int]]:
    """Returns (updated_rows, missing_rows, final_last_valid_payload_size)."""
    if tp not in currents:
        return 0, 0, None

    voltage_v, i_min_a, i_max_a = currents[tp]

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0, 0, None

    header = rows[0]
    idx_payload = header.index("payload") if "payload" in header else None
    if "payload_size_bytes" in header:
        idx_ps = header.index("payload_size_bytes")
    elif "payload_size" in header:
        idx_ps = header.index("payload_size")
    else:
        idx_ps = None

    idx_min = ensure_col(header, "energy_per_packet_min_mj")
    idx_max = ensure_col(header, "energy_per_packet_max_mj")

    updated_rows = 0
    missing_rows = 0
    last_valid_payload_size = initial_last_valid_payload_size

    for i in range(1, len(rows)):
        row = rows[i]
        while len(row) < len(header):
            row.append("")

        payload = row[idx_payload].strip() if idx_payload is not None and idx_payload < len(row) else ""
        if payload.startswith("CFG "):
            row[idx_min] = ""
            row[idx_max] = ""
            continue

        payload_size_from_row = None
        if idx_ps is not None and idx_ps < len(row):
            payload_size_from_row = parse_int(row[idx_ps].strip())

        # PACKET_LOST rows have payload_size 0; use previous successful packet's size
        payload_size = (
            payload_size_from_row
            if (payload_size_from_row is not None and payload_size_from_row > 0)
            else last_valid_payload_size
        )

        if payload_size is None:
            row[idx_min] = ""
            row[idx_max] = ""
            missing_rows += 1
            continue

        airtime_ms = get_airtime_ms(airtimes, bw, payload_size, sf)
        if airtime_ms is None:
            row[idx_min] = ""
            row[idx_max] = ""
            missing_rows += 1
            continue

        # E[J] = V * I * T[s]  => E[mJ] = V * I * T[ms]
        e_min_mj = voltage_v * i_min_a * airtime_ms
        e_max_mj = voltage_v * i_max_a * airtime_ms
        row[idx_min] = f"{e_min_mj:.2f}"
        row[idx_max] = f"{e_max_mj:.2f}"
        updated_rows += 1
        if payload_size_from_row is not None and payload_size_from_row > 0:
            last_valid_payload_size = payload_size_from_row

    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)

    return updated_rows, missing_rows, last_valid_payload_size


def main():
    currents = load_currents(CURRENT_FILE)
    airtimes = load_airtime(AIRTIME_FILE)

    # Collect all CSV paths in deterministic order so we can pass last_valid across files
    all_paths: list[tuple[str, int, int, int]] = []
    for root in DATA_ROOTS:
        for dirpath, _, filenames in os.walk(root):
            for fn in sorted(filenames):
                cfg = cfg_from_filename(fn)
                if cfg is None:
                    continue
                sf, bw, tp = cfg
                path = os.path.join(dirpath, fn)
                all_paths.append((path, sf, bw, tp))

    files_updated = 0
    rows_updated = 0
    rows_missing = 0
    last_valid_payload_size: Optional[int] = None

    for path, sf, bw, tp in all_paths:
        upd, miss, last_valid_payload_size = update_file(
            path, sf, bw, tp, currents, airtimes, last_valid_payload_size
        )
        if upd > 0 or miss > 0:
            files_updated += 1
            rows_updated += upd
            rows_missing += miss

    print(f"Done. Files touched: {files_updated}")
    print(f"Rows updated with new energy: {rows_updated}")
    print(f"Rows with missing lookup (left blank): {rows_missing}")


if __name__ == "__main__":
    main()
