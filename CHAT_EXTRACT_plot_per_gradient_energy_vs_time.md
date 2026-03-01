# Chat Extract: plot_per_gradient_energy_vs_time.py

Summary of changes made during this session for transferring to another machine.

## File Modified
- `tools/plots/plot_per_gradient_energy_vs_time.py`

## Key Features Implemented

### 1. Config Switch Markers (Letter Labels)
- **Labels**: B0-B3 (BW), S0-S5 (SF), T0-T2 (TP) - index-based
- **Style**: White text, bold, black border (patheffects.withStroke)
- **Legend**: Simple mapping "Config switch: SF7-12 -> S0-S5, BW62.5-500 -> B0-B3, TP2-22 -> T0-T2" (only visible params)

### 2. Partner Dashed Lines
- **TP** -> points to earlier BW and SF switches
- **BW** -> points to earlier SF switch
- **SF** -> no lines (cannot point)
- Lines connect to actual earlier config switch locations in the plot

### 3. Subtext Under Main Switcher
- **SF markers**: Show BW and TP underneath (smaller text)
- **BW markers**: Show TP only underneath (not SF)
- **TP markers**: No subtext (TP never has others switched simultaneously)

### 4. Thresholds
- `time_min_threshold=30` (min)
- `energy_min_threshold=60` (mJ)
- Config switches only shown when time > 30 min OR energy > 60 mJ

### 5. PER Label
- "PER (%)" label added above the gradient/colorbar scale on all three plots
- Function: `_add_per_label_above_colorbars(fig, n_cbars)`

### 6. Legend
- Compact legend with handlelength=0, handletextpad=0
- Position: bbox_to_anchor=(0, 0.88)

## Data Structure
- Config switch points: `(time_min, avg_energy, param_type, param_val, sf, bw, tp)` - full config included for partner lookup

## Plot Types
- **BW plot**: include_params=["sf", "tp"], plot_param="bw"
- **SF plot**: include_params=["bw", "tp"], plot_param="sf"
- **TP plot**: include_params=["sf", "bw"], plot_param="tp"

## Run Command
```bash
python tools/plots/plot_per_gradient_energy_vs_time.py
```

## Output Files
- `results/raw_test_data_plots/raw_per_gradient_energy_vs_time_bw_transitions.png`
- `results/raw_test_data_plots/raw_per_gradient_energy_vs_time_sf_transitions.png`
- `results/raw_test_data_plots/raw_per_gradient_energy_vs_time_tp_transitions.png`
