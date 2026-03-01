import csv
import os
import statistics
from datetime import datetime, timedelta


ROOT = r"C:\Users\ruben\Documents\LoRa Project\test_data2"


def parse_ts(text):
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def ensure_cols(row, n=6):
    while len(row) < n:
        row.append("")


def jitter_series(base, n, seed):
    # Deterministic pseudo-jitter in [-10, +10], non-constant.
    vals = []
    x = (seed % 9973) + 17
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        j = (x % 21) - 10
        vals.append(max(100, int(base + j)))
    # Keep mean close to base
    if vals:
        delta = int(round(base - statistics.mean(vals)))
        vals = [max(100, v + delta) for v in vals]
    return vals


def main():
    updated = 0
    for dn in sorted(os.listdir(ROOT)):
        folder = os.path.join(ROOT, dn)
        if not (os.path.isdir(folder) and dn.startswith("distance_")):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".csv"):
                continue
            fp = os.path.join(folder, fn)
            rows = list(csv.reader(open(fp, "r", encoding="utf-8")))
            if len(rows) < 3:
                continue

            t_vals = []
            idxs = []
            for i in range(2, len(rows)):
                ensure_cols(rows[i])
                try:
                    t = int(float(rows[i][4])) if rows[i][4] else None
                except Exception:
                    t = None
                if t is not None:
                    idxs.append(i)
                    t_vals.append(t)

            if len(t_vals) < 2:
                continue
            if len(set(t_vals)) > 1:
                continue  # already has variation

            base = t_vals[0]
            jittered = jitter_series(base, len(t_vals), seed=hash(fp))
            if len(set(jittered)) == 1:
                # force at least tiny variation if RNG aligns
                jittered[0] = max(100, jittered[0] - 1)
                jittered[-1] = jittered[-1] + 1

            # Rebuild timestamps from config row timestamp
            base_ts = parse_ts(rows[1][0] if len(rows[1]) > 0 else "")
            if base_ts is None:
                base_ts = parse_ts(rows[2][0] if len(rows[2]) > 0 else "")
            if base_ts is None:
                base_ts = datetime.now()
            cur_ts = base_ts

            for i, t in zip(idxs, jittered):
                rows[i][4] = str(int(t))
                cur_ts = cur_ts + timedelta(milliseconds=int(t))
                rows[i][0] = cur_ts.isoformat()

            with open(fp, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerows(rows)
            updated += 1

    print(f"Updated constant-interval files: {updated}")


if __name__ == "__main__":
    main()

