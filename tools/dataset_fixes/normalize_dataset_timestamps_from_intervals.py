import csv
import os
from datetime import datetime, timedelta


ROOT = r"C:\Users\ruben\Documents\LoRa Project\dataset"


def parse_dt(value):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def parse_ms(value):
    try:
        return float(str(value).strip())
    except Exception:
        return None


def main():
    files_updated = 0
    rows_updated = 0
    files_skipped = 0

    for walk_root, _, files in os.walk(ROOT):
        for name in files:
            if not name.endswith(".csv"):
                continue
            path = os.path.join(walk_root, name)
            rows = list(csv.reader(open(path, "r", encoding="utf-8-sig")))
            if len(rows) < 4:
                files_skipped += 1
                continue

            header = rows[0]
            if "timestamp" not in header or "time_between_messages_ms" not in header:
                files_skipped += 1
                continue

            ts_idx = header.index("timestamp")
            dt_idx = header.index("time_between_messages_ms")

            # Rule: row 3 (index 2) is the first trusted timestamp anchor.
            anchor = rows[2][ts_idx] if len(rows[2]) > ts_idx else ""
            current_ts = parse_dt(anchor)
            if current_ts is None:
                files_skipped += 1
                continue

            file_row_updates = 0
            for i in range(3, len(rows)):
                row = rows[i]
                if len(row) <= max(ts_idx, dt_idx):
                    continue
                interval_ms = parse_ms(row[dt_idx])
                if interval_ms is None:
                    continue
                current_ts = current_ts + timedelta(milliseconds=interval_ms)
                new_ts = current_ts.isoformat(timespec="microseconds")
                if row[ts_idx] != new_ts:
                    row[ts_idx] = new_ts
                    file_row_updates += 1

            if file_row_updates:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    csv.writer(f).writerows(rows)
                files_updated += 1
                rows_updated += file_row_updates

    print(f"Files updated: {files_updated}")
    print(f"Rows updated: {rows_updated}")
    print(f"Files skipped: {files_skipped}")


if __name__ == "__main__":
    main()

