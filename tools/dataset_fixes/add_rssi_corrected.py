"""
Add rssi_corrected column to raw_test_data and dataset CSVs.
Per SX1276 datasheet: RSSI values > -100 dBm do not follow linearity.
Correction: RSSI_corrected = 16/15 * (RSSI + 157) - 157
"""
import csv
import os

WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
DATA_ROOTS = [
    os.path.join(WORKSPACE, "raw_test_data"),
    os.path.join(WORKSPACE, "dataset"),
]


def rssi_corrected(rssi_val: float) -> float:
    """Apply SX1276 non-linearity correction for RSSI > -100 dBm."""
    return (16 / 15) * (rssi_val + 157) - 157


def parse_float(s):
    try:
        return float(s.strip()) if s else None
    except (ValueError, TypeError):
        return None


def update_file(path: str) -> bool:
    """Add rssi_corrected column. Returns True if file was modified."""
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return False
    header = rows[0]
    if "rssi_corrected" in header:
        return False  # already has it
    if "rssi" not in header:
        return False
    rssi_idx = header.index("rssi")
    insert_idx = rssi_idx + 1
    header.insert(insert_idx, "rssi_corrected")
    changed = False
    for i in range(1, len(rows)):
        row = rows[i]
        # Pad row so we can safely insert at insert_idx
        while len(row) < insert_idx:
            row.append("")
        rssi = parse_float(row[rssi_idx]) if rssi_idx < len(row) else None
        if rssi is not None:
            corr = rssi_corrected(rssi)
            new_val = f"{corr:.4f}"
        else:
            new_val = ""
        row.insert(insert_idx, new_val)
        changed = True
    if changed:
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
    return changed


def main():
    total = 0
    for root in DATA_ROOTS:
        if not os.path.isdir(root):
            continue
        for dn, subdirs, files in os.walk(root):
            for fn in files:
                if not fn.lower().endswith(".csv"):
                    continue
                path = os.path.join(dn, fn)
                if update_file(path):
                    total += 1
    print(f"Added rssi_corrected to {total} files")


if __name__ == "__main__":
    main()
