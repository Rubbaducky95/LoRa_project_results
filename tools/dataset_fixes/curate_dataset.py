#!/usr/bin/env python3
"""Sync the final curated dataset from raw_test_data into dataset/."""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
RAW_ROOT = WORKSPACE / "raw_test_data"
DATASET_ROOT = WORKSPACE / "dataset"
AIRTIME_SOURCE = WORKSPACE / "airtime_by_sf_bw_payload.csv"
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"

DISTANCE_RE = re.compile(r"^distance_(?P<distance>[0-9.]+)m$")
FILE_RE = re.compile(r"^SF(?P<sf>\d+)_BW(?P<bw>\d+)_TP(?P<tp>\d+)\.csv$")


def _iter_measurement_files():
    for path in sorted(RAW_ROOT.glob("distance_*m/SF*/SF*_BW*_TP*.csv")):
        distance_match = DISTANCE_RE.match(path.parts[-3])
        file_match = FILE_RE.match(path.name)
        if distance_match is None or file_match is None:
            continue
        yield {
            "source_path": path,
            "relative_path": path.relative_to(RAW_ROOT),
            "distance_m": float(distance_match.group("distance")),
            "sf": int(file_match.group("sf")),
            "bw_hz": int(file_match.group("bw")),
            "tp_dbm": int(file_match.group("tp")),
        }


def _reset_measurement_tree():
    DATASET_ROOT.mkdir(exist_ok=True)
    for path in DATASET_ROOT.glob("distance_*m"):
        if path.is_dir():
            shutil.rmtree(path)


def _copy_measurements(rows):
    for row in rows:
        target = DATASET_ROOT / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["source_path"], target)


def _copy_airtime_table():
    if AIRTIME_SOURCE.exists():
        shutil.copy2(AIRTIME_SOURCE, DATASET_ROOT / AIRTIME_SOURCE.name)


def _write_manifest(rows):
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "distance_m", "sf", "bw_hz", "tp_dbm"])
        for row in rows:
            writer.writerow(
                [
                    row["relative_path"].as_posix(),
                    f'{row["distance_m"]:.2f}'.rstrip("0").rstrip("."),
                    row["sf"],
                    row["bw_hz"],
                    row["tp_dbm"],
                ]
            )


def main():
    if not RAW_ROOT.is_dir():
        raise FileNotFoundError(f"Missing source dataset: {RAW_ROOT}")

    rows = sorted(
        _iter_measurement_files(),
        key=lambda row: (row["distance_m"], row["sf"], row["bw_hz"], row["tp_dbm"]),
    )
    if not rows:
        raise RuntimeError(f"No measurement files found under {RAW_ROOT}")

    _reset_measurement_tree()
    _copy_measurements(rows)
    _copy_airtime_table()
    _write_manifest(rows)

    distance_count = len({row["distance_m"] for row in rows})
    sf_count = len({row["sf"] for row in rows})
    bw_count = len({row["bw_hz"] for row in rows})
    tp_count = len({row["tp_dbm"] for row in rows})
    print(f"Curated dataset written to: {DATASET_ROOT}")
    print(f"Measurement files: {len(rows)}")
    print(f"Distances: {distance_count}")
    print(f"Spreading factors: {sf_count}")
    print(f"Bandwidths: {bw_count}")
    print(f"Transmit powers: {tp_count}")


if __name__ == "__main__":
    main()
