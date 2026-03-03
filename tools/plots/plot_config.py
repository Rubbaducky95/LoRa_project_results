"""
Global plotting configuration for IEEE-style figures.
Import these values in all plotting scripts for consistency.
"""

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
