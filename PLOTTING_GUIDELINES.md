# Plotting Guidelines

These rules apply to all plots in this project.

- Read `PROJECT_DESCRIPTION.md` first for project context, and use `Research_Work.pdf` for deeper paper context when needed, before changing or creating plots.
- Use the same font size everywhere in a figure, including axis labels, tick values, legends, annotations, and colorbar values. Use the shared IEEE values from `tools/plots/plot_config.py`.
- Use only IEEE plot widths from `tools/plots/plot_config.py`. Allowed widths are one-column width or two-column width. Do not introduce custom widths.
- Keep labels short. Prefer concise figure text and explain longer details in the paper body or caption.
- Any parameter with a unit must show the unit in parentheses immediately after the label, for example `Distance (m)` or `RSSI (dBm)`.
- Export plots as both `.png` and vector-based `.pdf` so figures remain readable in the paper. Prefer the shared `save_plot_outputs(...)` helper in `tools/plots/plot_config.py`.
- Every visual encoding must be explained in the figure itself. Lines, points, curves, markers, colors, fills, and gradients must have a legend, direct label, or clearly attached annotation.
