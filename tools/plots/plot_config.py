"""
Global plotting configuration for IEEE-style figures.
Import these values in all plotting scripts for consistency.
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
# Two-column figure (full width including column gap)
FIGSIZE_TWO_COL = (TEXTWIDTH_TWO_COL, FIGURE_HEIGHT)

# IEEE double-column (7.16" x 4" per How-to-plot)
FIGSIZE_IEEE_DOUBLE = (TEXTWIDTH_TWO_COL, 4)

# Single-column figure
FIGSIZE_ONE_COL = (TEXTWIDTH_ONE_COL, FIGURE_HEIGHT)

# Two-column with small colorbar (width_ratios=[1, 0.05]) - total = 3.5" (single column)
FIGSIZE_TWO_COL_CBAR_SMALL = (TEXTWIDTH_ONE_COL, FIGURE_HEIGHT)

# Two-column with larger colorbar panel (width_ratios=[1, 0.12]) - total = 3.5" (single column)
FIGSIZE_TWO_COL_CBAR_LARGE = (TEXTWIDTH_ONE_COL, FIGURE_HEIGHT)

# DPI for saving figures
SAVE_DPI = 220


def save_plot_outputs(fig, output_png, dpi=SAVE_DPI, bbox_inches="tight", save_pdf=True, **savefig_kwargs):
    """Save a figure as PNG and, by default, a sidecar PDF with the same basename."""
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
