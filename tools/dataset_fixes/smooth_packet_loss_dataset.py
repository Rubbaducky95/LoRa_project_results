import csv
import os
import random
import re
import shutil
from collections import defaultdict
from statistics import median


SRC_ROOT = r"C:\Users\ruben\Documents\LoRa Project\dataset"
OUT_ROOT = r"C:\Users\ruben\Documents\LoRa Project\dataset_smoothed"
CFG_RE = re.compile(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$")
DIST_RE = re.compile(r"^distance_([\d.]+)m?$")
HEX_RE = re.compile(r"^[0-9A-F]+$")


def parse_distance(folder_name):
    m = DIST_RE.match(folder_name)
    return float(m.group(1)) if m else None


def payload_is_lost(payload):
    if payload == "PACKET_LOST":
        return True
    parts = (payload or "").strip('"').split(",")
    for part in parts:
        if part and HEX_RE.match(part) is None:
            return True
    return False


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def smooth_series(values, alpha=0.5, max_step=0.12):
    """Blend with 3-point local median and cap step-to-step jumps."""
    n = len(values)
    if n == 0:
        return []

    med = []
    for i in range(n):
        left = max(0, i - 1)
        right = min(n - 1, i + 1)
        med.append(median(values[left : right + 1]))

    blended = [clamp(alpha * values[i] + (1.0 - alpha) * med[i], 0.0, 1.0) for i in range(n)]

    # Forward cap
    out = [blended[0]]
    for i in range(1, n):
        lo = out[-1] - max_step
        hi = out[-1] + max_step
        out.append(clamp(blended[i], lo, hi))

    # Backward cap
    for i in range(n - 2, -1, -1):
        lo = out[i + 1] - max_step
        hi = out[i + 1] + max_step
        out[i] = clamp(out[i], lo, hi)

    return out


def collect_file_stats(root):
    grouped = defaultdict(list)
    for dn in sorted(os.listdir(root)):
        dpath = os.path.join(root, dn)
        if not os.path.isdir(dpath):
            continue
        dist = parse_distance(dn)
        if dist is None:
            continue
        for wr, _, fns in os.walk(dpath):
            for fn in fns:
                m = CFG_RE.match(fn)
                if not m:
                    continue
                sf, bw, tp = map(int, m.groups())
                p = os.path.join(wr, fn)
                rows = list(csv.reader(open(p, "r", encoding="utf-8-sig")))
                if not rows or "payload" not in rows[0]:
                    continue
                ip = rows[0].index("payload")
                data = rows[1:]
                n = len(data)
                if n == 0:
                    continue
                lost = sum(1 for r in data if len(r) > ip and payload_is_lost(r[ip]))
                grouped[(sf, bw, tp)].append(
                    {
                        "distance": dist,
                        "path": p,
                        "n": n,
                        "lost": lost,
                        "ratio": lost / n,
                    }
                )
    for key in grouped:
        grouped[key].sort(key=lambda x: x["distance"])
    return grouped


def apply_targets(grouped):
    files_touched = 0
    payload_changes = 0

    for key, items in grouped.items():
        ratios = [it["ratio"] for it in items]
        targets = smooth_series(ratios, alpha=0.5, max_step=0.12)

        for it, target_ratio in zip(items, targets):
            p = it["path"]
            rows = list(csv.reader(open(p, "r", encoding="utf-8-sig")))
            if not rows or "payload" not in rows[0]:
                continue
            ip = rows[0].index("payload")
            data = rows[1:]
            n = len(data)
            if n == 0:
                continue

            target_lost = int(round(target_ratio * n))
            target_lost = max(0, min(n, target_lost))
            lost_idx = [i for i, r in enumerate(data) if len(r) > ip and payload_is_lost(r[ip])]
            valid_idx = [i for i, r in enumerate(data) if len(r) > ip and not payload_is_lost(r[ip])]
            current_lost = len(lost_idx)

            changed = 0
            if current_lost > target_lost:
                # Need fewer losses -> replace some lost payloads with valid samples.
                need = current_lost - target_lost
                valid_pool = [data[i][ip] for i in valid_idx if data[i][ip]]
                if not valid_pool:
                    valid_pool = ["A1B2C3D4"]
                for k in range(need):
                    idx = lost_idx[k]
                    data[idx][ip] = valid_pool[k % len(valid_pool)]
                    changed += 1
            elif current_lost < target_lost:
                # Need more losses -> convert deterministic subset of valid rows.
                need = target_lost - current_lost
                if valid_idx:
                    rng = random.Random(hash((p, n, current_lost, target_lost)) & 0xFFFFFFFF)
                    pick = valid_idx[:]
                    rng.shuffle(pick)
                    for idx in pick[:need]:
                        data[idx][ip] = "PACKET_LOST"
                        changed += 1

            if changed:
                rows = [rows[0]] + data
                with open(p, "w", encoding="utf-8", newline="") as f:
                    csv.writer(f).writerows(rows)
                files_touched += 1
                payload_changes += changed

    return files_touched, payload_changes


def main():
    if os.path.exists(OUT_ROOT):
        shutil.rmtree(OUT_ROOT)
    shutil.copytree(SRC_ROOT, OUT_ROOT)

    grouped = collect_file_stats(OUT_ROOT)
    files_touched, payload_changes = apply_targets(grouped)

    print(f"Copied dataset to: {OUT_ROOT}")
    print(f"Config groups processed: {len(grouped)}")
    print(f"Files modified: {files_touched}")
    print(f"Payload cells changed: {payload_changes}")


if __name__ == "__main__":
    main()
