import re
from pathlib import Path
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd

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
OUTPUT_PATH = Path(r"C:\Users\elias\Desktop\final_results_thesis\plots\minor_parameters")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

LOG_NAMES = ["Sepsis", "road_traffic_fine", "hospital_billing"]
NOISE_TYPES = ["base", "remove_enabled", "remove_events", "add_enabled", "add_events_no_trans"]

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

def plot_mean_f1_group(df_dict, petrify_df, imf_df, include_headings=True):

    config_keys = list(df_dict.keys())
    all_approaches = config_keys + ["Petrify", "IMf"]
    color_map = assign_colors(all_approaches)

    color_petrify = color_map.get("Petrify", "black")
    color_IMf = color_map.get("IMf", "red")

    # 1x2 Grid for IMftf (ax1) and IMfts (ax2)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)

    for config_key, df in df_dict.items():
        df = df.sort_values("threshold")
        color = color_map[config_key]
        z_order = 20 if config_key == "No Heuristics" else None

        # --- AX1: IMtf ---
        ax1.plot(
            df["threshold"],
            df["f_1_score_tf"],
            linestyle="-",
            marker="^",
            color=color,
            zorder=z_order
        )
        ax1.plot(
            df["threshold"],
            df["translucent_f_1_score_tf"],
            linestyle=":",
            marker="^",
            color=color,
            zorder=z_order
        )

        # --- AX2: IMts ---
        ax2.plot(
            df["threshold"],
            df["f_1_score_ts"],
            linestyle="-",
            marker="o",
            markersize=5,
            color=color,
            zorder=z_order
        )
        ax2.plot(
            df["threshold"],
            df["translucent_f_1_score_ts"],
            linestyle=":",
            marker="o",
            markersize=5,
            color=color,
            zorder=z_order
        )

    # --- IMf Baseline ---
    if not imf_df.empty:
        imf_df = imf_df.sort_values("threshold")
        # ax1 (tf)
        ax1.plot(
            imf_df["threshold"],
            imf_df["f_1_score"],
            linestyle="-",
            marker="x",
            color=color_IMf
        )
        ax1.plot(
            imf_df["threshold"],
            imf_df["translucent_f_1_score"],
            linestyle=":",
            marker="x",
            color=color_IMf
        )

        # ax2 (ts)
        ax2.plot(
            imf_df["threshold"],
            imf_df["f_1_score"],
            linestyle="-",
            marker="x",
            color=color_IMf
        )
        ax2.plot(
            imf_df["threshold"],
            imf_df["translucent_f_1_score"],
            linestyle=":",
            marker="x",
            color=color_IMf
        )

    # --- Petrify Baseline ---
    if not petrify_df.empty:
        petrify_normal_val = petrify_df["f_1_score"].iloc[0]
        petrify_trans_val = petrify_df["translucent_f_1_score"].iloc[0]

        # ax1
        ax1.plot([0, 1], [petrify_normal_val, petrify_normal_val], linestyle="-", color=color_petrify, zorder=0)
        ax1.plot([0, 1], [petrify_trans_val, petrify_trans_val], linestyle=":", color=color_petrify, zorder=0)

        # ax2
        ax2.plot([0, 1], [petrify_normal_val, petrify_normal_val], linestyle="-", color=color_petrify, zorder=0)
        ax2.plot([0, 1], [petrify_trans_val, petrify_trans_val], linestyle=":", color=color_petrify, zorder=0)

    # ------------------------------------------------------------------
    # Titles & Labels
    # ------------------------------------------------------------------
    ax1.set_xlabel("Filter Threshold")
    ax2.set_xlabel("Filter Threshold")
    ax1.set_ylabel("Mean F1-score")
    
    ax1.set_title("IMftf")
    ax2.set_title("IMfts")

    if include_headings:
        fig.suptitle("Mean F1-score | Across Datasets & Noise Types", fontsize=16)

    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # ------------------------------------------------------------------
    # Custom Legend
    # ------------------------------------------------------------------
    legend_elements = []

    for key in config_keys:
        legend_elements.append(
            Line2D([0], [0], color=color_map[key], lw=3, label=key)
        )

    legend_elements.append(Line2D([0], [0], color=color_petrify, lw=2, label="Petrify"))
    legend_elements.append(Line2D([0], [0], color=color_IMf, marker="x", lw=2, label="IMf"))

    legend_elements += [
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="Standard Metric"),
        Line2D([0], [0], color="black", lw=2, linestyle=":", label="Transl. Metric"),
        Line2D([0], [0], color="black", lw=2, marker="^", label="IMftf"),
        Line2D([0], [0], color="black", lw=2, marker="o", label="IMfts"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.08), 
        ncol=7, 
        frameon=True
    )

    plt.tight_layout()
    top_margin = 0.88 if include_headings else 0.93
    fig.subplots_adjust(top=top_margin, bottom=0.25) 
    
    output_file = OUTPUT_PATH / "mean_f1_score.pdf"
    fig.savefig(output_file, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close()

# ------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------

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
    return " | ".join(result.keys()).replace("_", " ")

# ------------------------------------------------------------------
# Main Routine
# ------------------------------------------------------------------

def visualize(include_headings=True):
    config_dfs = {}
    petrify_dfs = []
    imf_dfs = []

    target_cols = ["f_1_score_tf", "f_1_score_ts", "translucent_f_1_score_tf", "translucent_f_1_score_ts"]

    for log_name in LOG_NAMES:
        for noise_type in NOISE_TYPES:

            results_folder = BASE_RESULTS_PATH / log_name / "results" / "translucent" / noise_type
            if not results_folder.exists():
                continue

            for file in results_folder.glob("IMf_results_*.csv"):
                config_key = prettify_config_label(file.stem.replace("IMf_results_", ""))
                df = pd.read_csv(file)
                
                # Set values < 0 (-1.0) to 0.0 for mean calculation
                for col in target_cols:
                    if col in df.columns:
                        df.loc[df[col] < 0, col] = 0.0

                if config_key not in config_dfs:
                    config_dfs[config_key] = []
                config_dfs[config_key].append(df[["threshold"] + target_cols])

            # Load Petrify
            petrify_path = BASE_RESULTS_PATH / log_name / "results" / "petrify" / noise_type / "petrify_results.csv"
            if petrify_path.exists():
                p_df = pd.read_csv(petrify_path)
                for col in ["f_1_score", "translucent_f_1_score"]:
                    if col in p_df.columns:
                        p_df.loc[p_df[col] < 0, col] = 0.0
                petrify_dfs.append(p_df[["f_1_score", "translucent_f_1_score"]])

            # Load IMf
            imf_path = BASE_RESULTS_PATH / log_name / "results" / "IMf" / noise_type / "IMf_results.csv"
            if imf_path.exists():
                i_df = pd.read_csv(imf_path)
                for col in ["f_1_score", "translucent_f_1_score"]:
                    if col in i_df.columns:
                        i_df.loc[i_df[col] < 0, col] = 0.0
                imf_dfs.append(i_df[["threshold", "f_1_score", "translucent_f_1_score"]])

    if not config_dfs:
        return

    # Aggregate means across datasets and noise types
    mean_config_dfs = {}
    for config_key, dfs in config_dfs.items():
        combined = pd.concat(dfs, ignore_index=True)
        mean_config_dfs[config_key] = combined.groupby("threshold", as_index=False).mean()

    mean_petrify_df = pd.concat(petrify_dfs, ignore_index=True).mean(numeric_only=True).to_frame().T if petrify_dfs else pd.DataFrame()
    mean_imf_df = pd.concat(imf_dfs, ignore_index=True).groupby("threshold", as_index=False).mean() if imf_dfs else pd.DataFrame()

    plot_mean_f1_group(
        mean_config_dfs,
        mean_petrify_df,
        mean_imf_df,
        include_headings=include_headings
    )

if __name__ == "__main__":
    visualize(include_headings=False)