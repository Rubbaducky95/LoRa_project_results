# Curated Dataset

This directory is the final curated dataset for the paper.

Source:
- Measurement files are synced from `raw_test_data/`.
- The airtime lookup table is copied from `airtime_by_sf_bw_payload.csv`.

Regeneration:
- Run `python3 tools/dataset_fixes/curate_dataset.py`

Structure:
- `distance_<meters>m/SF<sf>/SF<sf>_BW<bw_hz>_TP<tp_dbm>.csv`
- `airtime_by_sf_bw_payload.csv`
- `manifest.csv`

Coverage:
- 16 distances from `6.25 m` to `100.0 m`
- 1008 measurement CSV files
- 6 spreading factors: `SF7` to `SF12`
- 4 bandwidth values overall: `62500 Hz`, `125000 Hz`, `250000 Hz`, `500000 Hz`
- 3 transmit powers: `2 dBm`, `12 dBm`, `22 dBm`

Notes:
- `manifest.csv` is the machine-readable index for the curated measurement files.
- Filenames encode `SF`, `BW (Hz)`, and `TP (dBm)`.
- `SF12` only appears with `BW 250000 Hz` and `BW 500000 Hz` in this dataset.
