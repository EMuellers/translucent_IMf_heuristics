import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import matplotlib as mpl
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
OUTPUT_PATH = Path(r"C:\eval_fall\new_eval\translucent_datasets\plots\delta_heur_10fold")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
NO_HEURISTICS_FILE_NAME = "IMf_10fold_translucent_self_loops_False_strict_end_activities_False_parallel_end_activities_heuristic_False_remove_arcs_heuristics_False_add_arcs_heuristics_False.csv"

LOG_NAMES = ["Sepsis", "hospital_billing", "road_traffic_fine"]
NOISE_TYPES = ["base", "remove_enabled", "remove_events", "add_enabled", "add_events_no_trans"]
#NOISE_TYPES = ["base"]
# Your custom abbreviation dictionary map
shorten_map = {
    "confidence": "conf.",
    "dependency score": "dep.",
    "support": "sup.",
    "parallel relationship frequency": "par.",
    "exclusive choice frequency": "excl.",
}

# ------------------------------------------------------------------
# Stable Color Assignment (tab20 colormap)
# ------------------------------------------------------------------

def assign_colors(config_keys):
    cmap = plt.get_cmap("tab20")
    colors = {}
    for i, key in enumerate(sorted(config_keys)):
        colors[key] = cmap(i % 20)
    return colors


# ------------------------------------------------------------------
# Plotting Function
# ------------------------------------------------------------------

def plot_metric_group(df_dict, log_name, noise_type, metric_group_name,
                      normal_cols, translucent_cols, ylabel, Petrify_df, IMf_df, IM_trans_no_heur_df,
                      include_headings=True):

    config_keys = list(df_dict.keys())
    all_approaches = config_keys + ["Petrify", "IMf", "IMf no heuristics"]
    color_map = assign_colors(all_approaches)
    
    color_petrify = color_map.get("Petrify", "black")
    color_IMf = color_map.get("IMf", "red")
    color_IMf_trans_no_heur = color_map.get("IMf no heuristics", "green")
    
    normal_col_non_trans = normal_cols[0][:-3]
    if translucent_cols:
        translucent_col_no_trans = translucent_cols[0][:-3]
    
    # Create a 1x2 grid side-by-side
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)

    for config_key, df in df_dict.items():
        # Aggregate across folds
        df_avg = df.groupby("threshold").mean().reset_index()
        df_avg = df_avg.sort_values("threshold")
        color = color_map[config_key]
        
        if config_key == "No Heuristics":
            z_order = 20
        else:
            z_order = None

        # --- AX1: IMtf ---
        mask1 = df_avg[normal_cols[0]] > 0
        ax1.plot(
            df_avg.loc[mask1, "threshold"],
            df_avg.loc[mask1, normal_cols[0]],
            linestyle="-",
            marker="^",
            color=color,
            zorder=z_order
        )

        if translucent_cols:
            mask2 = df_avg[translucent_cols[0]] > 0
            ax1.plot(
                df_avg.loc[mask2, "threshold"],
                df_avg.loc[mask2, translucent_cols[0]],
                linestyle=":",
                marker="^",
                color=color,
                zorder=z_order
            )

        # --- AX2: IMts ---
        mask3 = df_avg[normal_cols[1]] > 0
        ax2.plot(
            df_avg.loc[mask3, "threshold"],
            df_avg.loc[mask3, normal_cols[1]],
            linestyle="-",
            marker="o",
            markersize=5,
            color=color,
            zorder=z_order
        )

        if translucent_cols:
            mask4 = df_avg[translucent_cols[1]] > 0
            ax2.plot(
                df_avg.loc[mask4, "threshold"],
                df_avg.loc[mask4, translucent_cols[1]],
                linestyle=":",
                marker="o",
                markersize=5,
                color=color,
                zorder=z_order
            )
    
    # Aggregate baseline IMf rows across folds
    IMf_df_avg = IMf_df.groupby("threshold").mean().reset_index().sort_values("threshold")
    
    # IMf normal and translucent for tf graph
    mask1 = IMf_df_avg[normal_col_non_trans] > 0
    ax1.plot(
        IMf_df_avg.loc[mask1, "threshold"],
        IMf_df_avg.loc[mask1, normal_col_non_trans],
        linestyle="-",
        marker="x",
        color=color_IMf,
        zorder=None
    )

    if translucent_cols:
        mask2 = IMf_df_avg[translucent_col_no_trans] > 0
        ax1.plot(
            IMf_df_avg.loc[mask2, "threshold"],
            IMf_df_avg.loc[mask2, translucent_col_no_trans],
            linestyle=":",
            marker="x",
            color=color_IMf,
            zorder=None
        )
        
    # IMf normal and translucent for ts graph
    mask3 = IMf_df_avg[normal_col_non_trans] > 0
    ax2.plot(
        IMf_df_avg.loc[mask3, "threshold"],
        IMf_df_avg.loc[mask3, normal_col_non_trans],
        linestyle="-",
        marker="x",
        color=color_IMf,
        zorder=None
    )

    if translucent_cols:
        mask4 = IMf_df_avg[translucent_col_no_trans] > 0
        ax2.plot(
            IMf_df_avg.loc[mask4, "threshold"],
            IMf_df_avg.loc[mask4, translucent_col_no_trans],
            linestyle=":",
            marker="x",
            color=color_IMf,
            zorder=None
        )
        
    # Extract the average values across all folds for Petrify baseline
    petrify_normal_val = Petrify_df[normal_col_non_trans].mean()
    if petrify_normal_val < 0:
        petrify_normal_val = 0
    
    if translucent_cols:
        petrify_trans_val = Petrify_df[translucent_col_no_trans].mean()
        if petrify_trans_val < 0:
            petrify_trans_val = 0

    ax1.plot(
        [0, 1], [petrify_normal_val, petrify_normal_val],
        linestyle="-", color=color_petrify, zorder=0
    )

    if translucent_cols:
        ax1.plot(
            [0, 1], [petrify_trans_val, petrify_trans_val],
            linestyle=":", color=color_petrify, zorder=0
        )
        
    ax2.plot(
        [0, 1], [petrify_normal_val, petrify_normal_val],
        linestyle="-", color=color_petrify, zorder=0
    )

    if translucent_cols:
        ax2.plot(
            [0, 1], [petrify_trans_val, petrify_trans_val],
            linestyle=":", color=color_petrify, zorder=0
        )
    
    # Aggregate "No Heuristics" baseline rows across folds
    IM_trans_no_heur_df_avg = IM_trans_no_heur_df.groupby("threshold").mean().reset_index().sort_values("threshold")
    
    # IMtf no heuristics normal and translucent
    mask1 = IM_trans_no_heur_df_avg[normal_cols[0]] > 0
    ax1.plot(
        IM_trans_no_heur_df_avg.loc[mask1, "threshold"],
        IM_trans_no_heur_df_avg.loc[mask1, normal_cols[0]],
        linestyle="-",
        marker="^",
        color=color_IMf_trans_no_heur,
        zorder=20
    )

    if translucent_cols:
        mask2 = IM_trans_no_heur_df_avg[translucent_cols[0]] > 0
        ax1.plot(
            IM_trans_no_heur_df_avg.loc[mask2, "threshold"],
            IM_trans_no_heur_df_avg.loc[mask2, translucent_cols[0]],
            linestyle=":",
            marker="^",
            color=color_IMf_trans_no_heur,
            zorder=20
        )

    # IMts no heuristics normal and translucent
    mask3 = IM_trans_no_heur_df_avg[normal_cols[1]] > 0
    ax2.plot(
        IM_trans_no_heur_df_avg.loc[mask3, "threshold"],
        IM_trans_no_heur_df_avg.loc[mask3, normal_cols[1]],
        linestyle="-",
        marker="o",
        markersize=4,
        color=color_IMf_trans_no_heur,
        zorder=20
    )

    if translucent_cols:
        mask4 = IM_trans_no_heur_df_avg[translucent_cols[1]] > 0
        ax2.plot(
            IM_trans_no_heur_df_avg.loc[mask4, "threshold"],
            IM_trans_no_heur_df_avg.loc[mask4, translucent_cols[1]],
            linestyle=":",
            marker="o",
            markersize=4,
            color=color_IMf_trans_no_heur,
            zorder=20
        )

    # Labels and grid setup
    ax1.set_xlabel("Filter Threshold")
    ax2.set_xlabel("Filter Threshold")
    ax1.set_ylabel(ylabel)
    
    ax1.set_title("IMftf")
    ax2.set_title("IMfts")

    if include_headings:
        noise_str = "No Noise" if noise_type == "base" else f"Noise: {noise_type}"
        fig.suptitle(f"{metric_group_name} (10-Fold CV) | Log: {log_name} | {noise_str}", fontsize=16)

    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # Custom Legend Construction
    legend_elements = []

    for key in config_keys:
        # Cleanly substitute long mapping values using the shorten_map loop
        display_label = key
        for long_name, short_name in shorten_map.items():
            display_label = display_label.replace(long_name, short_name)
        legend_elements.append(
            Line2D([0], [0], color=color_map[key], lw=3, label=display_label)
        )
    
    legend_elements.append(Line2D([0], [0], color=color_petrify, lw=3, label="Petrify"))
    legend_elements.append(Line2D([0], [0], color=color_IMf, lw=3, marker="x", label="IMf"))
    legend_elements.append(Line2D([0], [0], color=color_IMf_trans_no_heur, lw=3, label="No Heuristics"))

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

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.058), 
        ncol=7, 
        frameon=True
    )

    plt.tight_layout()
    top_margin = 0.88 if include_headings else 0.93
    fig.subplots_adjust(top=top_margin, bottom=0.25) 
    
    output_file = OUTPUT_PATH / f"{log_name}_{noise_type}_{metric_group_name}_10fold.pdf"
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
                / "translucent_delta_10fold"
                / noise_type
            )

            if not results_folder.exists():
                continue

            df_dict = {}

            for file in results_folder.glob("IMf_10fold_translucent_*.csv"):
                config_key = file.stem.replace("IMf_10fold_", "")
                config_key = prettify_config_label(config_key)
                df = pd.read_csv(file)
                df_dict[config_key] = df

            if not df_dict:
                continue
            
            log_name_pretty = prettify_log_name(log_name)
            noise_type_pretty = prettify_noise_type(noise_type)

            petrify_results_path = (
                BASE_RESULTS_PATH
                / log_name
                / "results"
                / "petrify_10fold"
                / noise_type
                / "petrify_10fold_results.csv"
            )
            petrify_df = pd.read_csv(petrify_results_path)
            
            IMf_results_path = (
                BASE_RESULTS_PATH
                / log_name
                / "results"
                / "IMf_10fold"
                / noise_type
                / "IMf_10fold_results.csv"
            )
            IMf_df = pd.read_csv(IMf_results_path)
            
            IMf_trans_base_path = (
                BASE_RESULTS_PATH
                / log_name
                / "results"
                / "translucent_minor_10fold"
                / noise_type
                / NO_HEURISTICS_FILE_NAME
            )
            IMf_trans_base_df = pd.read_csv(IMf_trans_base_path)
            
            # Fitness & Translucent Fitness
            plot_metric_group(
                df_dict, log_name_pretty, noise_type_pretty, "Fitness",
                ["fitness_tf", "fitness_ts"], ["translucent_fitness_tf", "translucent_fitness_ts"],
                "Fitness", petrify_df, IMf_df, IMf_trans_base_df, include_headings=include_headings
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
        "remove_arcs_heuristics": "remove_arcs",
        "add_arcs_heuristics": "add_arcs",
    }
    result = {rename_map.get(k, k): v for k, v in result.items()}
    result = {k: v for k, v in result.items() if (v != 'False' and v != "True")}
    
    result_string = " | ".join(result.values()).replace("_", " ")

    return result_string

if __name__ == "__main__":
    visualize(include_headings=False)