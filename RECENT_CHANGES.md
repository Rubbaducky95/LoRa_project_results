# Recent Changes

## 2026-03-02

### Energy vs Time Plots (v3)
- **Config path uses actual switch times**: Path x-coordinates now use `tm` from `config_change_T_init.csv` instead of mean packet time
- **BW plot path**: Full 63-config sequence (SF7-BW62.5-TP2 → TP12 → TP22 → BW125-TP2 → ... → SF8 → ...)
- **SF plot path**: Full 63-config sequence (same as BW)
- **TP plot path**: Simplified 24-point path (one point per SF,BW pair: SF7→all BWs→SF8→all BWs→...)
