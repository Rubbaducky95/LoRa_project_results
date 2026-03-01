import csv
import math
import os
import random
import re
import statistics
from collections import defaultdict

import fill_from_raw_test_data as base


ROOT = r"C:\Users\ruben\Documents\LoRa Project\test_data2"


def is_hex_payload(payload_str):
    if not payload_str:
        return False
    parts = payload_str.strip('"').split(",")
    for part in parts:
        if not part:
            continue
        if re.match(r"^[0-9A-F]+$", part) is None:
            return False
    return True


def parse_cfg_from_filename(filename):
    m = re.match(r"^SF(\d+)_BW(\d+)_TP(\d+)\.csv$", filename)
    if not m:
        return None
    return tuple(map(int, m.groups()))


def parse_distance(folder_name):
    m = re.match(r"^distance_([\d.]+)m?$", folder_name)
    return float(m.group(1)) if m else None


def build_raw_loss_model():
    by_cfg = defaultdict(list)
    for folder_name, folder_path, _ in base.list_distance_folders(base.RAW_DIR):
        d = parse_distance(folder_name)
        if d is None:
            continue
        for fn in os.listdir(folder_path):
            cfg = parse_cfg_from_filename(fn)
            if not cfg:
                continue
            sf, bw, tp = cfg
            if tp not in (2, 12, 22):
                continue
            fp = os.path.join(folder_path, fn)
            try:
                rows = list(csv.reader(open(fp, "r", encoding="utf-8")))
            except Exception:
                continue
            total = 0
            bad = 0
            for r in rows[2:]:
                if len(r) < 6:
                    continue
                total += 1
                if not is_hex_payload(r[5]):
                    bad += 1
            if total == 0:
                continue
            by_cfg[(sf, bw, tp)].append((d, (bad * 100.0) / total))
    for k in list(by_cfg.keys()):
        by_cfg[k].sort(key=lambda x: x[0])
    return by_cfg


def estimate_loss(distance, cfg, raw_loss):
    sf, bw, tp = cfg
    pts = raw_loss.get((sf, bw, tp), [])
    if not pts:
        # fallback with expected tendency
        base_loss = max(0.0, (distance - 50.0) * 0.08)
        if tp == 2:
            base_loss *= 1.3
        elif tp == 22:
            base_loss *= 0.8
        if sf >= 11:
            base_loss *= 0.85
        elif sf <= 7:
            base_loss *= 1.15
        return min(35.0, base_loss)

    if len(pts) == 1:
        d0, l0 = pts[0]
        if distance <= d0:
            return max(0.0, l0)
        return min(35.0, max(0.0, l0 + 0.05 * (distance - d0)))

    if distance <= pts[0][0]:
        return max(0.0, pts[0][1])
    if distance >= pts[-1][0]:
        d1, l1 = pts[-2]
        d2, l2 = pts[-1]
        slope = (l2 - l1) / (d2 - d1) if d2 != d1 else 0.05
        if slope <= 0:
            slope = 0.05
        return min(35.0, max(0.0, l2 + slope * (distance - d2)))

    for i in range(len(pts) - 1):
        d1, l1 = pts[i]
        d2, l2 = pts[i + 1]
        if d1 <= distance <= d2 and d2 != d1:
            ratio = (distance - d1) / (d2 - d1)
            return max(0.0, min(35.0, l1 + (l2 - l1) * ratio))
    return max(0.0, min(35.0, pts[-1][1]))


def make_corrupted_payload(valid_payload):
    parts = valid_payload.strip('"').split(",")
    if not parts:
        parts = ["1000000", "100", "8", "AA", "BB", "CC", "DD", "EE"]
    n = len(parts)
    edits = random.randint(2, min(4, n))
    idxs = random.sample(range(n), edits)
    bad_chars = ["<", ">", "$", "p", "g", "L", "G", "?", "!", "&"]

    for idx in idxs:
        part = parts[idx] if parts[idx] else "0"
        pos = random.randint(0, len(part))
        ch = random.choice(bad_chars)
        if pos >= len(part):
            part = part + ch
        else:
            part = part[:pos] + ch + part[pos + 1 :]
        parts[idx] = part
    return ",".join(parts)


def enforce_file(fp, distance, cfg, time_model, raw_loss):
    rows = list(csv.reader(open(fp, "r", encoding="utf-8")))
    if len(rows) < 3:
        return False
    if "[GENERATED_FROM_RAW]" not in rows[1][5]:
        return False

    sf, bw, tp = cfg
    target_time_mean, _ = base.estimate_time(distance, sf, bw, time_model)
    target_loss = estimate_loss(distance, cfg, raw_loss)

    # Normalize RSSI formatting and collect current mean time.
    times = []
    valid_indices = []
    for i in range(2, len(rows)):
        r = rows[i]
        if len(r) < 6:
            continue
        # RSSI must be floored double (X.0)
        try:
            rf = float(r[1])
        except Exception:
            rf = -90.0
        r_floor = int(math.floor(rf))
        r[1] = f"{r_floor}.0"

        try:
            t = float(r[4]) if r[4] else 0.0
            if t > 0:
                times.append(t)
        except Exception:
            pass

        # reset payload candidates list (we'll reapply loss deterministically)
        valid_indices.append(i)

    # Adjust time intervals so per-file mean matches raw-based target.
    current_mean = statistics.mean(times) if times else target_time_mean
    scale = (target_time_mean / current_mean) if current_mean > 0 else 1.0
    for i in range(2, len(rows)):
        r = rows[i]
        if len(r) < 6:
            continue
        try:
            t = float(r[4]) if r[4] else target_time_mean
        except Exception:
            t = target_time_mean
        new_t = int(max(100, round(t * scale)))
        r[4] = str(new_t)

    # Reset payloads to valid-like form if currently invalid placeholder.
    for i in range(2, len(rows)):
        r = rows[i]
        if len(r) < 6:
            continue
        if not is_hex_payload(r[5]):
            # regenerate valid baseline payload
            ctr = 1000000 + (i - 1)
            hex_values = ",".join([f"{random.randint(0,255):02X}" for _ in range(8)])
            r[5] = f"{ctr},100,8,{hex_values}"

    # Reapply target invalid payload count.
    total = max(0, len(rows) - 2)
    target_bad = int(round(total * (target_loss / 100.0)))
    random.seed(hash(fp) % (2 ** 32))
    bad_idxs = set(random.sample(valid_indices, min(target_bad, len(valid_indices))))
    for i in valid_indices:
        if i in bad_idxs:
            rows[i][5] = make_corrupted_payload(rows[i][5])

    with open(fp, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    return True


def main():
    base.RAW_DIR = r"C:\Users\ruben\Documents\LoRa Project\raw_test_data"
    time_model = base.build_raw_models()[1]
    raw_loss = build_raw_loss_model()

    updated = 0
    for folder in sorted(os.listdir(ROOT)):
        folder_path = os.path.join(ROOT, folder)
        if not (os.path.isdir(folder_path) and folder.startswith("distance_")):
            continue
        distance = parse_distance(folder)
        if distance is None:
            continue

        for fn in os.listdir(folder_path):
            cfg = parse_cfg_from_filename(fn)
            if not cfg:
                continue
            fp = os.path.join(folder_path, fn)
            if enforce_file(fp, distance, cfg, time_model, raw_loss):
                updated += 1

    print(f"Updated generated files: {updated}")


if __name__ == "__main__":
    main()

