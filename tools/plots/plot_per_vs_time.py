"""
Plot PER vs time only, for all BW, all SF, and all TP (3 separate plots).
Same style as the time panel in plot_per_vs_multiple_configs.py.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow importing from same directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from plot_per_vs_multiple_configs import (
    COLORS,
    DATA_ROOT,
    MARKERS,
    WORKSPACE,
    collect_packets_for_time_plot,
    fit_curve_log,
    format_config_label,
    parse_config_arg,
    parse_filter_spec,
    setup_plot_style,
)


def main():
    setup_plot_style()
    output_dir = os.path.join(WORKSPACE, "results", "raw_test_data_plots")
    os.makedirs(output_dir, exist_ok=True)

    for config in ["bw", "sf", "tp"]:
        dims = parse_config_arg(config)
        filters = []
        time_by_cfg = collect_packets_for_time_plot(DATA_ROOT, dims, filters)
        configs = sorted(time_by_cfg.keys())
        color_map = {c: COLORS[i % len(COLORS)] for i, c in enumerate(configs)}
        marker_map = {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(configs)}

        fig, ax = plt.subplots(figsize=(8, 5))
        for cfg in configs:
            buckets = time_by_cfg.get(cfg, {})
            xs = []
            ys = []
            for min_bucket in sorted(buckets.keys()):
                lost, total = buckets[min_bucket]
                if total > 0:
                    per_pct = 100.0 * lost / total
                    if per_pct > 0:
                        xs.append(min_bucket)
                        ys.append(per_pct)
            if not xs:
                continue
            ax.scatter(
                xs, ys,
                s=40,
                c=[color_map[cfg]],
                marker=marker_map[cfg],
                alpha=0.9,
                edgecolors="black",
                linewidths=0.5,
            )
            curve = fit_curve_log(np.array(xs), np.array(ys))
            if curve is not None:
                ax.plot(curve[0], curve[1], color=color_map[cfg], linestyle="-", linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Time (minutes, internal timer from first packet)")
        ax.set_ylabel("PER (%)")
        ax.set_title(f"PER vs time (by {config.upper()})")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(0.01, 100)

        handles = [
            plt.Line2D(
                [0], [0],
                marker=marker_map[c],
                color="w",
                markerfacecolor=color_map[c],
                markersize=8,
                label=format_config_label(dims, c),
            )
            for c in configs
        ]
        ncol = min(len(configs), 8)
        fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=7, bbox_to_anchor=(0.5, -0.08))

        fig.tight_layout()
        out_path = os.path.join(output_dir, f"raw_per_vs_time_{config}.png")
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Saved: raw_per_vs_time_{config}.png")


if __name__ == "__main__":
    main()
