import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import re

# ------------------------------------------------------------------
# Global Styling (Thesis-ready)
# ------------------------------------------------------------------

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 9,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2,
})

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

BASE_RESULTS_PATH = Path(r"C:\eval_fall\new_eval\translucent_datasets")
OUTPUT_PATH = Path(r"C:\eval_fall\new_eval\translucent_datasets\plots\minor_parameters")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

LOG_NAMES = ["Sepsis", "road_traffic_fine", "hospital_billing"] 
#LOG_NAMES = ["road_traffic_fine"]
#NOISE_TYPES = ["base"]
NOISE_TYPES = ["base","remove_enabled", "remove_events", "add_enabled", "add_events_no_trans"]

# ------------------------------------------------------------------
# Stable Color Assignment (tab20 colormap)
# ------------------------------------------------------------------

def assign_colors(config_keys):
    cmap = plt.get_cmap("tab20")
    #cmap = plt.get_cmap("Dark2")
    colors = {}
    for i, key in enumerate(sorted(config_keys)):
        colors[key] = cmap(i % 20) # set to 20 for tab20, 8 for dark2 (only 8 distinct colors)
    return colors


# ------------------------------------------------------------------
# Plotting Function
# ------------------------------------------------------------------

def plot_metric_group(df_dict, log_name, noise_type, metric_group_name,
                      normal_cols, translucent_cols, ylabel, Petrify_df, IMf_df,
                      include_headings=True):

    config_keys = list(df_dict.keys())
    all_approaches = config_keys + ["Petrify", "IMf"]
    color_map = assign_colors(all_approaches)

    color_petrify = color_map.get("Petrify", "black")
    color_IMf = color_map.get("IMf", "red")

    normal_col_non_trans = normal_cols[0][:-3]
    if translucent_cols:
        translucent_col_no_trans = translucent_cols[0][:-3]

    # 1. Create a 1x2 grid. We use sharey=True so the Y-axis scale matches perfectly side-by-side
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)

    for config_key, df in df_dict.items():
        df = df.sort_values("threshold")
        color = color_map[config_key]
        
        if config_key == "No Heuristics":
            z_order = 20  # Draw "No Heuristics" on top
        else:
            z_order = None

        # --- AX1: IMtf ---
        # 1. IMtf normal
        if metric_group_name != "No. of Fallthroughs":  # Fallthrough count doesn't have translucent metrics, so we skip the normal metric for better visibility
            mask1 = df[normal_cols[0]] > 0
        else:
            mask1 = df[normal_cols[0]] >= 0
        ax1.plot(
            df.loc[mask1, "threshold"],
            df.loc[mask1, normal_cols[0]],
            linestyle="-",
            marker="^",
            color=color,
            zorder=z_order
        )

        # 2. IMtf translucent
        if translucent_cols:
            mask2 = df[translucent_cols[0]] > 0
            ax1.plot(
                df.loc[mask2, "threshold"],
                df.loc[mask2, translucent_cols[0]],
                linestyle=":",
                marker="^",
                color=color,
                zorder=z_order
            )

        # --- AX2: IMts ---
        # 3. IMts normal
        if metric_group_name != "No. of Fallthroughs": 
            mask3 = df[normal_cols[1]] > 0
        else:
            mask3 = df[normal_cols[1]] >= 0
        ax2.plot(
            df.loc[mask3, "threshold"],
            df.loc[mask3, normal_cols[1]],
            linestyle="-",
            marker="o",
            markersize=5,
            color=color,
            zorder=z_order
        )

        # 4. IMts translucent
        if translucent_cols:
            mask4 = df[translucent_cols[1]] > 0
            ax2.plot(
                df.loc[mask4, "threshold"],
                df.loc[mask4, translucent_cols[1]],
                linestyle=":",
                marker="o",
                markersize=5,
                color=color,
                zorder=z_order
            )

    z_order = None

    # CRITICAL FIX: Explicitly sort IMf_df by threshold to prevent out-of-order line tracking
    IMf_df = IMf_df.sort_values("threshold")

    # 5. IMf normal for tf graph
    if metric_group_name != "No. of Fallthroughs":
        mask1 = IMf_df[normal_col_non_trans] > 0
    else:
        mask1 = IMf_df[normal_col_non_trans] >= 0
    ax1.plot(
        IMf_df.loc[mask1, "threshold"],
        IMf_df.loc[mask1, normal_col_non_trans],
        linestyle="-",
        marker="x",
        color=color_IMf,
        zorder=z_order
    )

    # 6. IMf translucent values for tf graph
    if translucent_cols:
        mask2 = IMf_df[translucent_col_no_trans] > 0
        ax1.plot(
            IMf_df.loc[mask2, "threshold"],
            IMf_df.loc[mask2, translucent_col_no_trans],
            linestyle=":",
            marker="x",
            color=color_IMf,
            zorder=z_order
        )
        
    # 7. IMf normal for ts graph
    if metric_group_name != "No. of Fallthroughs":
        mask3 = IMf_df[normal_col_non_trans] > 0
    else:
        mask3 = IMf_df[normal_col_non_trans] >= 0
    ax2.plot(
        IMf_df.loc[mask3, "threshold"],
        IMf_df.loc[mask3, normal_col_non_trans],
        linestyle="-",
        marker="x",
        color=color_IMf,
        zorder=z_order
    )

    # 8. IMf translucent values for ts graph
    if translucent_cols:
        mask4 = IMf_df[translucent_col_no_trans] > 0
        ax2.plot(
            IMf_df.loc[mask4, "threshold"],
            IMf_df.loc[mask4, translucent_col_no_trans],
            linestyle=":",
            marker="x",
            color=color_IMf,
            zorder=z_order
        )

    # 9-12. Petrify normal and translucent graphs (Excluded from "No. of Fallthroughs")
    if metric_group_name != "No. of Fallthroughs":
        petrify_normal_val = Petrify_df[normal_col_non_trans].iloc[0]
        if petrify_normal_val < 0:
            petrify_normal_val = 0
        
        if translucent_cols:
            petrify_trans_val = Petrify_df[translucent_col_no_trans].iloc[0]
            if petrify_trans_val < 0:
                petrify_trans_val = 0

        # 9. Petrify normal for tf graph
        ax1.plot(
            [0, 1],
            [petrify_normal_val, petrify_normal_val],
            linestyle="-",
            color=color_petrify,
            zorder=0
        )

        # 10. Petrify translucent values for tf graph
        if translucent_cols:
            ax1.plot(
                [0, 1],
                [petrify_trans_val, petrify_trans_val],
                linestyle=":",
                color=color_petrify,
                zorder=0
            )
            
        # 11. Petrify normal for ts graph
        ax2.plot(
            [0, 1],
            [petrify_normal_val, petrify_normal_val],
            linestyle="-",
            markersize=4,
            color=color_petrify,
            zorder=0
        )

        # 12. Petrify translucent values for ts graph
        if translucent_cols:
            ax2.plot(
                [0, 1],
                [petrify_trans_val, petrify_trans_val],
                linestyle=":",
                markersize=4,
                color=color_petrify,
                zorder=0
            )

    # ------------------------------------------------------------------
    # Titles & Labels
    # ------------------------------------------------------------------
    ax1.set_xlabel("Filter Threshold")
    ax2.set_xlabel("Filter Threshold")
    ax1.set_ylabel(ylabel)
    
    # Subplot columns are always titled
    ax1.set_title("IMftf")
    ax2.set_title("IMfts")

    if include_headings:
        # Set the main top figure title
        noise_str = "No Noise" if noise_type == "base" else f"Noise: {noise_type}"
        fig.suptitle(f"{metric_group_name} | Log: {log_name} | {noise_str}", fontsize=16)

    if metric_group_name == "No. of Fallthroughs":
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # ------------------------------------------------------------------
    # Custom Legend (Bottom Placement)
    # ------------------------------------------------------------------

    legend_elements = []

    # Parameter sets (colors)
    for key in config_keys:
        legend_elements.append(
            Line2D([0], [0], color=color_map[key], lw=3, label=key)
        )

    # Standalone approaches (colors)
    columns = 6
    if metric_group_name != "No. of Fallthroughs":
        columns = 7
        legend_elements.append(Line2D([0], [0], color=color_petrify, lw=2, label="Petrify"))
    legend_elements.append(Line2D([0], [0], color=color_IMf, marker="x", lw=2, label="IMf"))

    # Metric types
    legend_elements += [
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="Normal Metric"),
    ]
    
    if translucent_cols:
        legend_elements += [
            Line2D([0], [0], color="black", lw=2, linestyle=":", label="Transl. Metric"),
        ]
        
    legend_elements += [
        Line2D([0], [0], color="black", lw=2, marker="^", label="IMftf"),
        Line2D([0], [0], color="black", lw=2, marker="o", label="IMfts"),
    ]

    # Attach legend to the figure, centered at the bottom.
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08), 
        ncol=columns, 
        frameon=True
    )

    # Adjust layout to prevent overlap, making room for the suptitle (top) and legend (bottom)
    plt.tight_layout()
    
    # Dynamically scale top margin depending on whether the overarching figure title is active
    top_margin = 0.88 if include_headings else 0.93
    fig.subplots_adjust(top=top_margin, bottom=0.25) 
    
    output_file = OUTPUT_PATH / f"{log_name}_{noise_type}_{metric_group_name}.pdf"
    fig.savefig(output_file, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close()


# ------------------------------------------------------------------
# Main Visualization Routine
# ------------------------------------------------------------------

def visualize(include_headings=True):

    for log_name in LOG_NAMES:
        for noise_type in NOISE_TYPES:

            results_folder = (
                BASE_RESULTS_PATH
                / log_name
                / "results"
                / "translucent"
                / noise_type
            )

            if not results_folder.exists():
                continue

            df_dict = {}

            for file in results_folder.glob("IMf_results_*.csv"):
                config_key = file.stem.replace("IMf_results_", "")
                config_key = prettify_config_label(config_key)
                df = pd.read_csv(file)
                df_dict[config_key] = df

            if not df_dict:
                continue
            
            #  Prettify Log Name and Noise Type for Titles
            log_name_pretty = prettify_log_name(log_name)
            noise_type_pretty = prettify_noise_type(noise_type)

            # Handle petrify and IMf paths
            petrify_results_path = (
                BASE_RESULTS_PATH
                / log_name
                / "results"
                / "petrify"
                / noise_type
                / "petrify_results.csv"
            )
            petrify_df = pd.read_csv(petrify_results_path)
            
            IMf_results_path = (
                BASE_RESULTS_PATH
                / log_name
                / "results"
                / "IMf"
                / noise_type
                / "IMf_results.csv"
            )
            IMf_df = pd.read_csv(IMf_results_path)

            # Fitness
            plot_metric_group(
                df_dict,
                log_name_pretty,
                noise_type_pretty,
                "Fitness",
                ["fitness_tf", "fitness_ts"],
                ["translucent_fitness_tf", "translucent_fitness_ts"],
                "Fitness",
                petrify_df,
                IMf_df,
                include_headings=include_headings
            )

            # Precision
            plot_metric_group(
                df_dict,
                log_name_pretty,
                noise_type_pretty,
                "Precision",
                ["precision_tf", "precision_ts"],
                ["translucent_precision_tf", "translucent_precision_ts"],
                "Precision",
                petrify_df,
                IMf_df,
                include_headings=include_headings
            )

            # F1 Score
            plot_metric_group(
                df_dict,
                log_name_pretty,
                noise_type_pretty,
                "F1-score",
                ["f_1_score_tf", "f_1_score_ts"],
                ["translucent_f_1_score_tf", "translucent_f_1_score_ts"],
                "F1-score",
                petrify_df,
                IMf_df,
                include_headings=include_headings
            )
            
            # Simplicity
            plot_metric_group(
                df_dict,
                log_name_pretty,
                noise_type_pretty,
                "Simplicity",
                ["simplicity_tf", "simplicity_ts"],
                None,  # No translucent metrics here!
                "Simplicity",
                petrify_df,
                IMf_df,
                include_headings=include_headings
            )
            
            # Fallthrough Count
            plot_metric_group(
                df_dict,
                log_name_pretty,
                noise_type_pretty,
                "No. of Fallthroughs",
                ["fallthrough_count_tf", "fallthrough_count_ts"],
                None,  # No translucent metrics here!
                "No. of fallthroughs",
                petrify_df,
                IMf_df,
                include_headings=include_headings
            )

def prettify_noise_type(noise_type: str) -> str:
    noise_map = {
        "base": "No Noise",
        "add_enabled": "Add Enabled Activities",
        "add_events": "Add Events",
        "remove_enabled": "Remove Enabled Activities",
        "change_events": "Change Events",
        "add_events_no_trans": "Add Events"
    }
    return noise_map.get(noise_type, noise_type.replace("_", " ").title())

def prettify_log_name(raw_name: str) -> str:
    name_map = {
        "Sepsis": "Sepsis",
        "hospital_billing": "Hospital Billing",
        "road_traffic_fine": "Road Traffic Fine"
    }
    return name_map.get(raw_name, raw_name.replace("_", " ").title())

def prettify_config_label(config_string: str) -> str:
    keywords = [
        'translucent_self_loops',
        'strict_end_activities',
        'parallel_end_activities_heuristic',
        'remove_arcs_heuristics',
        'add_arcs_heuristics',
    ]
    pattern = r'(?:_)(' + '|'.join(re.escape(k) for k in keywords) + r')'
    
    parts = re.split(pattern, config_string)
    
    result = {}
    first_key, first_value = parts[0].rsplit('_', 1)
    result[first_key] = first_value.lstrip('_')

    for i in range(1, len(parts), 2):
        key = parts[i]
        value = parts[i + 1].lstrip('_')
        result[key] = value

    result = {k: v for k, v in result.items() if v != 'False'}
    if not result:
        return "No Heuristics"
    
    rename_map = {
        "translucent_self_loops": "self-loops",
        "strict_end_activities": "strict-end",
        "parallel_end_activities_heuristic": "parallel-end",
        "remove_arcs_heuristics": "remove_arcs",
        "add_arcs_heuristics": "add_arcs",
    }
    result = {rename_map.get(k, k): v for k, v in result.items()}
    result_string = " | ".join(result.keys()).replace("_", " ")

    return result_string

if __name__ == "__main__":
    visualize(include_headings=False)