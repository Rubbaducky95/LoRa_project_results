"""
Global plotting configuration for IEEE-style figures.

Usage rules for this repository:
- A figure with one plot/panel should use `FIGSIZE_ONE_COL`.
- Use `FIGSIZE_TWO_COL` or `FIGSIZE_IEEE_DOUBLE` only for multi-panel figures
  or when a one-column layout is demonstrably too cramped.
- During iteration, export PNG only. Enable PDF explicitly for final figure
  versions by passing `save_pdf=True` to `save_plot_outputs(...)`.
"""

import os

# Font sizes (IEEE 10pt body text)
IEEE_FONTSIZE = 10

# Standard figure height for IEEE conference papers
FIGURE_HEIGHT = 2.5  # inches - consistent across single and two-column

# Text widths for reference
TEXTWIDTH_ONE_COL = 3.5   # inches
TEXTWIDTH_TWO_COL = 7.16  # inches (includes column gap)

# Figure sizes (inches)
# Multi-panel or extra-dense two-column figure (full width including column gap)
FIGSIZE_TWO_COL = (TEXTWIDTH_TWO_COL, FIGURE_HEIGHT)

# IEEE double-column tall figure (7.16" x 4"); reserve for multi-panel 3D or similarly dense layouts
FIGSIZE_IEEE_DOUBLE = (TEXTWIDTH_TWO_COL, 4)

# Default for any single-panel figure
FIGSIZE_ONE_COL = (TEXTWIDTH_ONE_COL, FIGURE_HEIGHT)

# Single-column width figure with a small colorbar allocation
FIGSIZE_TWO_COL_CBAR_SMALL = (TEXTWIDTH_ONE_COL, FIGURE_HEIGHT)

# Single-column width figure with a larger colorbar allocation
FIGSIZE_TWO_COL_CBAR_LARGE = (TEXTWIDTH_ONE_COL, FIGURE_HEIGHT)

# DPI for saving figures
SAVE_DPI = 220

# Iteration exports should stay PNG-only; pass save_pdf=True for final paper-ready outputs.
SAVE_PDF_BY_DEFAULT = False


def save_plot_outputs(fig, output_png, dpi=SAVE_DPI, bbox_inches="tight", save_pdf=SAVE_PDF_BY_DEFAULT, **savefig_kwargs):
    """Save PNG by default and optionally a sidecar PDF for final versions."""
    if not output_png:
        raise ValueError("output_png must be a non-empty path.")

    png_path = output_png if output_png.lower().endswith(".png") else f"{output_png}.png"
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"

    out_dir = os.path.dirname(png_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches, **savefig_kwargs)
    if save_pdf:
        fig.savefig(pdf_path, bbox_inches=bbox_inches, **savefig_kwargs)

    return png_path, pdf_path if save_pdf else None
