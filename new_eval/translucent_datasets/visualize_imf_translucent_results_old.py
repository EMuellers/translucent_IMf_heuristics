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

BASE_RESULTS_PATH = Path(r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\translucent_datasets")
OUTPUT_PATH = Path(r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\translucent_datasets\plots\minor_parameters")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

#LOG_NAMES = ["Sepsis", "hospital_billing", "road_traffic_fine"]
LOG_NAMES = ["hospital_billing"]
#NOISE_TYPES = ["base", "remove_enabled"]
NOISE_TYPES = ["remove_enabled"]

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
                      normal_cols, translucent_cols, ylabel):

    config_keys = list(df_dict.keys())
    color_map = assign_colors(config_keys)

    fig, ax = plt.subplots(figsize=(9, 6))
    """
    for config_key, df in df_dict.items():
        df = df.sort_values("threshold")
        color = color_map[config_key]
        
        if config_key == "No heuristics":
            z_order = 20  # Draw "No heuristics" on top
        else:
            z_order = None

        # IMtf normal
        ax.plot(
            df["threshold"],
            df[normal_cols[0]],
            linestyle="-",
            marker="^",
            color=color,
            zorder=z_order
        )

        # IMtf translucent
        ax.plot(
            df["threshold"],
            df[translucent_cols[0]],
            linestyle=":",
            marker="^",
            color=color,
            zorder=z_order
        )

        # IMts normal (marker differentiates variant)
        ax.plot(
            df["threshold"],
            df[normal_cols[1]],
            linestyle="-",
            marker="o",
            markersize=4,
            color=color,
            zorder=z_order
        )

        # IMts translucent
        ax.plot(
            df["threshold"],
            df[translucent_cols[1]],
            linestyle=":",
            marker="o",
            markersize=4,
            color=color,
            zorder=z_order
        )
    """
    for config_key, df in df_dict.items():
        df = df.sort_values("threshold")
        color = color_map[config_key]
        
        if config_key == "No heuristics":
            z_order = 20  # Draw "No heuristics" on top
        else:
            z_order = None

        # 1. IMtf normal
        mask1 = df[normal_cols[0]] > 0
        ax.plot(
            df.loc[mask1, "threshold"],
            df.loc[mask1, normal_cols[0]],
            linestyle="-",
            marker="^",
            color=color,
            zorder=z_order
        )

        # 2. IMtf translucent
        mask2 = df[translucent_cols[0]] > 0
        ax.plot(
            df.loc[mask2, "threshold"],
            df.loc[mask2, translucent_cols[0]],
            linestyle=":",
            marker="^",
            color=color,
            zorder=z_order
        )

        # 3. IMts normal (marker differentiates variant)
        mask3 = df[normal_cols[1]] > 0
        ax.plot(
            df.loc[mask3, "threshold"],
            df.loc[mask3, normal_cols[1]],
            linestyle="-",
            marker="o",
            markersize=4,
            color=color,
            zorder=z_order
        )

        # 4. IMts translucent
        mask4 = df[translucent_cols[1]] > 0
        ax.plot(
            df.loc[mask4, "threshold"],
            df.loc[mask4, translucent_cols[1]],
            linestyle=":",
            marker="o",
            markersize=4,
            color=color,
            zorder=z_order
        )

    ax.set_xlabel("Filter Threshold")
    ax.set_ylabel(ylabel)
    if noise_type == "base":        
        ax.set_title(f"{metric_group_name}\nLog: {log_name} | No Noise")
    else:
        ax.set_title(f"{metric_group_name}\nLog: {log_name} | Noise: {noise_type}")
    ax.grid(True, alpha=0.3)
    

    # ------------------------------------------------------------------
    # Custom Legend (clean & readable)
    # ------------------------------------------------------------------

    legend_elements = []

    # Parameter sets (colors)
    for key in config_keys:
        legend_elements.append(
            Line2D([0], [0], color=color_map[key], lw=3, label=key)
        )

    # Metric types
    legend_elements += [
        Line2D([0], [0], color="black", lw=2, linestyle="-",
               label="Normal Metric"),
        Line2D([0], [0], color="black", lw=2, linestyle=":",
               label="Translucent Metric"),
        Line2D([0], [0], color="black", lw=2, marker="^",
               label="IMtf"),
        Line2D([0], [0], color="black", lw=2, marker="o",
               label="IMts"),
    ]

    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=True
    )

    plt.tight_layout()
    output_file = OUTPUT_PATH / f"{log_name}_{noise_type}_{metric_group_name}.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------
# Main Visualization Routine
# ------------------------------------------------------------------

def visualize():

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
            log_name = prettify_log_name(log_name)
            noise_type = prettify_noise_type(noise_type)

            # Fitness
            plot_metric_group(
                df_dict,
                log_name,
                noise_type,
                "Fitness",
                ["fitness_tf", "fitness_ts"],
                ["translucent_fitness_tf", "translucent_fitness_ts"],
                "Fitness"
            )

            # Precision
            plot_metric_group(
                df_dict,
                log_name,
                noise_type,
                "Precision",
                ["precision_tf", "precision_ts"],
                ["translucent_precision_tf", "translucent_precision_ts"],
                "Precision"
            )

            # F1 Score
            plot_metric_group(
                df_dict,
                log_name,
                noise_type,
                "F1 Score",
                ["f_1_score_tf", "f_1_score_ts"],
                ["translucent_f_1_score_tf", "translucent_f_1_score_ts"],
                "F1 Score"
            )

def get_best_average_f1_per_threshold():

    results_summary = []

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

            # ----------------------------------------------------------
            # Load all configs into a single dataframe
            # ----------------------------------------------------------

            all_data = []

            for file in results_folder.glob("IMf_results_*.csv"):
                config_key = file.stem.replace("IMf_results_", "")
                df = pd.read_csv(file)
                df["config"] = config_key
                all_data.append(df)

            if not all_data:
                continue

            full_df = pd.concat(all_data, ignore_index=True)

            # ----------------------------------------------------------
            # Compute per-threshold best configs
            # ----------------------------------------------------------

            for threshold in sorted(full_df["threshold"].unique()):

                df_f = full_df[full_df["threshold"] == threshold]

                # --- IMtf normal F1 ---
                tf_normal = (
                    df_f.groupby("config")["f_1_score_tf"]
                    .mean()
                    .reset_index()
                )
                best_tf_normal = tf_normal.loc[tf_normal["f_1_score_tf"].idxmax()]

                # --- IMtf translucent F1 ---
                tf_trans = (
                    df_f.groupby("config")["translucent_f_1_score_tf"]
                    .mean()
                    .reset_index()
                )
                best_tf_trans = tf_trans.loc[
                    tf_trans["translucent_f_1_score_tf"].idxmax()
                ]

                # --- IMts normal F1 ---
                ts_normal = (
                    df_f.groupby("config")["f_1_score_ts"]
                    .mean()
                    .reset_index()
                )
                best_ts_normal = ts_normal.loc[ts_normal["f_1_score_ts"].idxmax()]

                # --- IMts translucent F1 ---
                ts_trans = (
                    df_f.groupby("config")["translucent_f_1_score_ts"]
                    .mean()
                    .reset_index()
                )
                best_ts_trans = ts_trans.loc[
                    ts_trans["translucent_f_1_score_ts"].idxmax()
                ]

                results_summary.append({
                    "log": log_name,
                    "noise": noise_type,
                    "threshold": threshold,

                    "IMtf_best_f1": best_tf_normal["f_1_score_tf"],
                    "IMtf_best_f1_config": prettify_config_label(best_tf_normal["config"]),

                    "IMtf_best_translucent_f1": best_tf_trans["translucent_f_1_score_tf"],
                    "IMtf_best_translucent_f1_config": prettify_config_label(best_tf_trans["config"]),

                    "IMts_best_f1": best_ts_normal["f_1_score_ts"],
                    "IMts_best_f1_config": prettify_config_label(best_ts_normal["config"]),

                    "IMts_best_translucent_f1": best_ts_trans["translucent_f_1_score_ts"],
                    "IMts_best_translucent_f1_config": prettify_config_label(best_ts_trans["config"]),
                })

    summary_df = pd.DataFrame(results_summary)

    return summary_df

def prettify_noise_type(noise_type: str) -> str:
    noise_map = {
        "base": "No Noise",
        "add_enabled": "Add Enabled Activities",
        "add_events": "Add Events",
        "remove_enabled": "Remove Enabled Activities",
        "change_events": "Change Events"
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
    """
    Converts raw config string into a readable label.
    - Removes parameters with value False
    - Prints parameter name if True
    - Prints param=value if non-boolean
    """

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
    result[first_key] = first_value.lstrip('_')  # remove leading underscore before True/False

    for i in range(1, len(parts), 2):
        key = parts[i]
        value = parts[i + 1].lstrip('_')  # remove leading underscore before True/False
        result[key] = value

    # Remove false entries, meaning heuristic was not applied
    result = {k: v for k, v in result.items() if v != 'False'}
    if not result:
        return "No heuristics"
    
    
    

    # optional shorter display names
    rename_map = {
        "translucent_self_loops": "self_loops",
        "strict_end_activities": "strict_end",
        "parallel_end_activities_heuristic": "parallel_end",
        "remove_arcs_heuristics": "remove_arcs",
        "add_arcs_heuristics": "add_arcs",
    }
    # Change the naming of entries
    result = {rename_map.get(k, k): v for k, v in result.items()}
    
    result_string = " | ".join(result.keys()).replace("_", " ")

    return result_string

if __name__ == "__main__":
    visualize()
    #summary_df = get_best_average_f1_per_threshold()
    #print(summary_df)
    #print("Done")