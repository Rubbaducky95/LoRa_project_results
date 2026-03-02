# LoRa Project Results - Project Description

A handover document for switching machines or sessions. Use this to understand the project without re-explaining everything.

---

## 1. Project Overview

Testing LoRa configurations to create a dataset for machine learning, and analysing the test data for further optimisation. The project measures PER (Packet Error Rate), RSSI, energy, and related metrics across SF (Spreading Factor), BW (Bandwidth), TP (Transmit Power), and distance. Data is organized by config and distance; scripts produce plots and analysis for a conference paper.

For more context on the research goals, see: `Research_Work.pdf`

---

## 2. Directory Structure


| Path                           | Purpose                                    |
| ------------------------------ | ------------------------------------------ |
| `dataset/`                     | Processed dataset with SF subfolders       |
| `dataset_smoothed/`            | Smoothed PER data from `dataset/`          |
| `original_test_data/`          | Raw receiver logs (flat, no SF subfolders) |
| `raw_test_data/`               | Raw receiver logs (may have SF subfolders) |
| `tools/`                       | Scripts for processing, analysis, plotting |
| `tools/dataset_fixes/`         | Dataset processing scripts                 |
| `tools/plots/`                 | Plotting scripts                           |
| `tools/analysis/`              | Analysis scripts                           |
| `results/`                     | Plots, QA reports, outputs                 |
| `results/raw_test_data_plots/` | Plots from raw data                        |
| `.cursor/rules/`               | Cursor rules (e.g. How-to-plot.mdc)        |


### External Paths (may need updating on new machine)

Scripts fall back to `C:\Users\ruben\Documents\LoRa Project\` when repo-relative paths fail. Update `WORKSPACE` / `DATA_ROOT` in scripts if the project moves.

---

## 3. Dataset Structure

### Folder Naming

- **Distance**: `distance_<value>m` (e.g. `distance_6.25m`, `distance_100.0m`)
- **Distances**: 6.25, 12.5, 18.75, 25.0, 31.25, 37.5, 43.75, 50.0, 56.25, 62.5, 68.75, 75.0, 81.25, 87.5, 93.75, 100.0 m

### File Layouts

**original_test_data** (flat):

- Path: `distance_<X>m/SF<#>_BW<#>_TP<#>.csv`
- No SF subfolder

**raw_test_data** / **dataset** (with SF subfolders):

- Path: `distance_<X>m/SF<#>/SF<#>_BW<#>_TP<#>.csv`

### CSV Columns (raw_test_data / dataset)


| Column                            | Description                                                              |
| --------------------------------- | ------------------------------------------------------------------------ |
| `payload`                         | Hex payload or `PACKET_LOST`                                             |
| `time_since_boot_ms`              | Time since receiver boot (ms)                                            |
| `time_since_transmission_init_ms` | Time since first packet of distance (ms) - used for energy vs time plots |
| `payload_size_bytes`              | Payload size                                                             |
| `rssi`                            | Received Signal Strength (dBm)                                           |
| `rssi_corrected`                  | Antenna-corrected RSSI                                                   |
| `kalman_rssi`                     | Kalman-filtered RSSI                                                     |
| `sma_rssi`                        | Simple moving average RSSI                                               |
| `energy_per_packet_min_mj`        | Min energy per packet (mJ)                                               |
| `energy_per_packet_max_mj`        | Max energy per packet (mJ)                                               |
| `tx_interval_ms`                  | Inter-packet interval (ms)                                               |
| `timestamp`                       | Packet timestamp                                                         |
| `snr_db`                          | SNR (dB) - optional, added by add_snr_to_raw_dataset.py                  |


### Payload Validity

- **Valid**: Comma-separated hex values (e.g. `257342,100,8,E9,E0,21,...`)
- **Invalid**: `PACKET_LOST` or non-hex
- **Config row**: `CFG sf=X sbw=Y tp=Z` - skipped in PER calculations

---

## 4. Configuration Parameters


| Parameter    | Values                        | Unit                         |
| ------------ | ----------------------------- | ---------------------------- |
| **SF**       | 7, 8, 9, 10, 11, 12           | -                            |
| **BW**       | 62500, 125000, 250000, 500000 | Hz (62.5, 125, 250, 500 kHz) |
| **TP**       | 2, 12, 22                     | dBm                          |
| **Distance** | 6.25 - 100                    | m                            |


### Config Order (test sequence)

SF first, then BW, then TP: SF7-12, for each BW 62.5-500 kHz, for each TP 2/12/22.

### Excluded Configs

SF11 and SF12 at 62.5 kHz are excluded in some scripts (fill_from_raw_test_data.py).

### Config Filename

Pattern: `SF<#>_BW<#>_TP<#>.csv` (e.g. `SF7_BW62500_TP12.csv`)

---

## 5. Key Auxiliary Files


| File                                                   | Purpose                                                                                                                |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `results/raw_test_data_plots/config_change_T_init.csv` | Config switch times for energy vs time plots. Columns: `packet_index`, `config` (SF,BW as "7, 62.5"), `TP`, `T_init_s` |
| `airtime_by_sf_bw_payload.csv`                         | LoRa airtime (ms) by SF, BW, payload size                                                                              |
| `results/qa/dataset_outlier_summary.json`              | Outlier detection summary                                                                                              |
| `results/qa/dataset_outlier_report.csv`                | Per-file outlier details                                                                                               |


---

## 6. Key Scripts

### Plots (`tools/plots/`)


| Script                                | Purpose                                                                                          |
| ------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `plot_per_gradient_energy_vs_time.py` | PER heatmap (energy vs time), config switch markers, letter labels (B0-B3, S0-S5, T0-T2)         |
| `plot_per_vs_multiple_configs.py`     | PER vs distance, multiple configs; defines WORKSPACE, DATA_ROOT, BW_VALUES, SF_VALUES, TP_VALUES |
| `plot_per_vs_distance_per_config.py`  | PER vs distance per config                                                                       |
| `plot_raw_per_vs_distance.py`         | Raw PER vs distance                                                                              |
| `plot_time_since_boot.py`             | Time since boot / transmission init                                                              |


### Dataset Fixes (`tools/dataset_fixes/`)


| Script                                               | Purpose                               |
| ---------------------------------------------------- | ------------------------------------- |
| `patch_time_since_boot_and_add_transmission_init.py` | Add `time_since_transmission_init_ms` |
| `add_rssi_corrected.py`                              | Add `rssi_corrected`                  |
| `add_snr_to_raw_dataset.py`                          | Add SNR column                        |
| `add_packet_energy_to_raw_dataset.py`                | Add energy columns                    |
| `detect_dataset_outliers.py`                         | Outlier detection                     |


---

## 7. Plot Conventions (`.cursor/rules/How-to-plot.mdc`)

- IEEE conference format (single or double column)
- Figure size: 7.16" x 4" (IEEE double-column)
- Font size 10pt, LaTeX-style text
- Bandwidth in kHz
- Axis labels with units
- No titles inside plots (handled in LaTeX)
- Thick lines and dots, no small textures

**Reference**: [IEEEtran HOWTO](https://mirror.accum.se/mirror/CTAN/macros/latex/contrib/IEEEtran/IEEEtran_HOWTO.pdf) - How to Use the IEEEtran LaTeX Class

---

## 8. Dependencies

- **matplotlib** - plotting
- **numpy** - numerical operations
- Standard library: csv, re, os, statistics

No requirements.txt in repo. Install: `pip install matplotlib numpy`

---

## 9. Path Resolution

Scripts in `tools/plots/` (e.g. plot_per_vs_multiple_configs.py):

1. Try `../raw_test_data` relative to script (works when repo is `LoRa_project_results`)
2. Fallback: `C:\Users\ruben\Documents\LoRa Project\raw_test_data`

`plot_per_gradient_energy_vs_time.py` imports from `plot_per_vs_multiple_configs` for DATA_ROOT, WORKSPACE, BW_VALUES, etc.

---

## 10. Energy vs Time Plots (plot_per_gradient_energy_vs_time.py)

- Uses `time_since_transmission_init_ms` when available (continuous across distances)
- Config switches from `config_change_T_init.csv`
- Letter labels: B0-B3 (BW), S0-S5 (SF), T0-T2 (TP)
- Dashed lines: TP->BW/SF, BW->SF (to earlier occurrences)
- Subtext under SF/BW: SF shows BW+TP; BW shows TP only
- Thresholds: time_min_threshold=30 min, energy_min_threshold=60 mJ
- PER label above colorbar

---

## 11. Quick Start (New Machine)

1. Clone/copy repo
2. Ensure `raw_test_data` or `dataset` exists at expected path; update WORKSPACE/DATA_ROOT if needed
3. `pip install matplotlib numpy`
4. Run plots: `python tools/plots/plot_per_gradient_energy_vs_time.py`
5. Outputs in `results/raw_test_data_plots/`

