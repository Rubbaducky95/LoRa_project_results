"""Count time_since_boot_ms values above 1 million across raw test data."""
import csv
import os
import re

WORKSPACE = r"C:\Users\ruben\Documents\LoRa Project"
RAW_ROOT = os.path.join(WORKSPACE, "raw_test_data")
CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")

count = 0
for root, _, files in os.walk(RAW_ROOT):
    for fn in files:
        if not CFG_RE.match(fn):
            continue
        path = os.path.join(root, fn)
        rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
        if not rows or "time_since_boot_ms" not in rows[0]:
            continue
        header = rows[0]
        time_idx = header.index("time_since_boot_ms")
        for r in rows[1:]:
            if time_idx < len(r) and r[time_idx]:
                try:
                    t = float(r[time_idx])
                    if t > 1_000_000:
                        count += 1
                except (ValueError, TypeError):
                    pass
print(f"time_since_boot_ms > 1,000,000: {count}")
