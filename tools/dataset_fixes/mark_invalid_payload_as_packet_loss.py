"""
Mark rows with invalid payload format as PACKET_LOST.
Valid payload: timestamp,100,8,XX,XX,XX,... where XX are 1-2 char hex bytes.
- Only allowed chars: 0-9, A-F, a-f, comma
- First value: time_since_boot_ms (digits)
- Second: 100, Third: 8
- Remaining: hex bytes (1-2 chars each, comma-separated)
Corrupted payloads (e.g. merged hex like 8F5A, invalid chars) are marked as PACKET_LOST.
"""
import csv
import os
import re

WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
DATA_ROOTS = [
    os.path.join(WORKSPACE, "raw_test_data"),
    os.path.join(WORKSPACE, "dataset"),
]

CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")
HEX_BYTE_RE = re.compile(r"^[0-9A-Fa-f]{1,2}$")
ALLOWED_CHARS_RE = re.compile(r"^[0-9A-Fa-f,]+$")


def payload_is_valid(payload: str) -> bool:
    """
    Valid format: timestamp,100,8,XX,XX,... where XX are 1-2 char hex bytes.
    """
    if not payload or not isinstance(payload, str):
        return False
    payload = payload.strip().strip('"')
    if not payload:
        return False
    # Only 0-9, A-F, a-f, comma
    if not ALLOWED_CHARS_RE.match(payload):
        return False
    parts = payload.split(",")
    if len(parts) < 4:
        return False
    # First: digits only (timestamp)
    if not parts[0].isdigit():
        return False
    # Second: 100, Third: 8
    if parts[1] != "100" or parts[2] != "8":
        return False
    # Remaining: each 1-2 hex chars
    for part in parts[3:]:
        if not part or not HEX_BYTE_RE.match(part):
            return False
    return True


def main():
    total_marked = 0
    files_touched = 0

    for root in DATA_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if CFG_RE.match(fn) is None:
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, "r", encoding="utf-8-sig", newline="") as f:
                    rows = list(csv.reader(f))
                if not rows:
                    continue

                header = rows[0]
                idx_payload = header.index("payload") if "payload" in header else None
                idx_ps = None
                if "payload_size_bytes" in header:
                    idx_ps = header.index("payload_size_bytes")
                elif "payload_size" in header:
                    idx_ps = header.index("payload_size")
                if idx_payload is None or idx_ps is None:
                    continue

                changed = False
                for i in range(1, len(rows)):
                    row = rows[i]
                    while len(row) < len(header):
                        row.append("")

                    payload = row[idx_payload].strip() if idx_payload < len(row) else ""
                    if payload.startswith("CFG ") or payload == "PACKET_LOST":
                        continue

                    if not payload_is_valid(payload):
                        row[idx_payload] = "PACKET_LOST"
                        if idx_ps < len(row):
                            row[idx_ps] = "0"
                        changed = True
                        total_marked += 1

                if changed:
                    with open(path, "w", encoding="utf-8", newline="") as f:
                        csv.writer(f).writerows(rows)
                    files_touched += 1

    print(f"Marked {total_marked} rows as PACKET_LOST across {files_touched} files.")


if __name__ == "__main__":
    main()
