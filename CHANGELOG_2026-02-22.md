# Summary of Changes – 2026-02-22

## 1. LoRa Airtime Formula Parameters

Updated `tools/dataset_fixes/regenerate_airtime_raw_lora.py` to match the SX1276 datasheet and your test setup:

| Parameter | Previous | Final |
|-----------|----------|-------|
| **Coding Rate (CR)** | 4/5 | **4/6** |
| **CRC** | ON | **OFF** |
| **Explicit Header** | ON | ON (unchanged) |
| **Preamble Length** | 8 | 8 (unchanged) |
| **Low Data Rate Optimization** | SF11/12 @ 125 kHz only | **T_sym > 16 ms** (automatic) |

### Low Data Rate Optimization (new rule)

Low data rate opt is now applied when symbol time exceeds 16 ms:

- **SF10 @ 62.5 kHz**: ON (16.4 ms)
- **SF11 @ 62.5 kHz**: ON (32.8 ms)
- **SF11 @ 125 kHz**: ON (16.4 ms)
- **SF12 @ 62.5 kHz**: ON (65.5 ms)
- **SF12 @ 125 kHz**: ON (32.8 ms)
- All other configs: OFF

---

## 2. Regenerated Airtime Table

- **File**: `airtime_by_sf_bw_payload.csv`
- **Method**: Ran `regenerate_airtime_raw_lora.py` with the new parameters
- **Coverage**: BW 62.5, 125, 250, 500 kHz; SF 7–12; payload sizes 34–39 bytes
- **Rows**: 144

---

## 3. Recomputed Energy Values

- **Script**: `tools/dataset_fixes/recompute_energy_minmax_from_currents_airtime.py`
- **Files updated**: 2,016
- **Rows updated**: 201,600
- **Missing lookups**: 0

Energy per packet (min/max mJ) was recalculated using the new airtime table.

---

## 4. Regenerated Plots

### Energy plots
- `results/raw_test_data_plots/raw_energy_minmax_gradient_by_tp_greyscale.png`
- `results/raw_test_data_plots/raw_energy_minmax_gradient_by_tp_color.png`

### PER plots
- `raw_per_vs_multiple_configs_sf.png`
- `raw_per_vs_multiple_configs_bw.png`
- `raw_per_vs_multiple_configs_tp.png`
- `raw_per_vs_multiple_configs_distance.png`
- `raw_per_vs_multiple_configs_sf_bw.png`
- `raw_per_vs_multiple_configs_sf_tp.png`
- `raw_per_vs_multiple_configs_tp_bw.png`
- `raw_per_vs_distance_aggregated.png`
- `raw_per_vs_distance_all_configs.png`
- `raw_per_vs_distance_all_labeled.png`

---

## 5. Airtime Formula Reference

```
T_sym = 2^SF / BW_khz  (ms)
T_preamble = (preamble_len + 4.25) × T_sym
payload_bit = 8×payload_bytes - 4×SF + 8 + (20 if explicit_header) + (16 if CRC)
bits_per_symbol = SF - 2  (if T_sym > 16 ms) else SF
payload_symbol = ceil(payload_bit / (4×bits_per_symbol)) × CR + 8
T_payload = payload_symbol × T_sym
T_packet = T_preamble + T_payload
```

---

---

## 6. RSSI Correction (SX1276 Non-Linearity)

Per SX1276 datasheet: RSSI values > -100 dBm do not follow linearity.

**Formula**: `RSSI_corrected = 16/15 * (RSSI + 157) - 157`

**Added**:
- New script `tools/dataset_fixes/add_rssi_corrected.py`
- `rssi_corrected` column added to all CSVs in `raw_test_data` and `dataset` (2,016 files)

**Updated plotting scripts** (now use `rssi_corrected` when available):
- `plot_per_vs_multiple_configs.py`
- `plot_per_vs_rssi.py`
- `plot_raw_avg_rssi_vs_distance_by_tp.py`
- `plot_avg_rssi_gradient_by_tp.py`
- `plot_rssi_family_generic.py` (default `--metric-col` changed to `rssi_corrected`)

**Regenerated plots**: PER plots and RSSI plots

---

## Files Modified

- `tools/dataset_fixes/regenerate_airtime_raw_lora.py` – parameter updates
- `tools/dataset_fixes/add_rssi_corrected.py` – new script
- `airtime_by_sf_bw_payload.csv` – regenerated
- All raw test data and dataset CSVs – energy columns updated, `rssi_corrected` added
- Plotting scripts – use `rssi_corrected` when available
- All plot PNGs in `results/raw_test_data_plots/` – regenerated
