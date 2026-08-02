import os
import itertools
import pandas as pd
import numpy as np

# --- Configuration ---
# Adjust this path to match your local folder structure.
# The script assumes files are nested as: CSV_BASE_DIR/{dataset_key}/{noise_key}/filenames.csv
CSV_BASE_DIR = r"C:\eval_fall\new_eval\translucent_datasets" 
OUTPUT_TXT_FILE = r"C:\Users\elias\OneDrive - Students RWTH Aachen University\Masterarbeit\thesis-main\appendix_eval_figures\figures_single.tex"
FIGURES_BASE_PATH = "./figures/eval/plots/minor_parameters"

# Structural mappings
datasets = {
    "Sepsis": "Sepsis",
    "road_traffic_fine": "Road Traffic Fine",
    "hospital_billing": "Hospital Billing"
}

noise_types = {
    "base": "No Noise",
    "remove_enabled": "Remove Enabled Activities",
    "remove_events": "Remove Events",
    "add_enabled": "Add Enabled Activities",
    "add_events_no_trans": "Add Events"
}

# The 5 core figure groups and their associated dataframe columns
metrics_config = {
    "Fitness": {
        "standard": ("fitness_tf", "fitness_ts"),
        "translucent": ("translucent_fitness_tf", "translucent_fitness_ts")
    },
    "Precision": {
        "standard": ("precision_tf", "precision_ts"),
        "translucent": ("translucent_precision_tf", "translucent_precision_ts")
    },
    "F1-score": {
        "standard": ("f_1_score_tf", "f_1_score_ts"),
        "translucent": ("translucent_f_1_score_tf", "translucent_f_1_score_ts")
    },
    "Simplicity": {
        "standard": ("simplicity_tf", "simplicity_ts")
    },
    "No. of Fallthroughs": {
        "standard": ("fallthrough_count_tf", "fallthrough_count_ts")
    }
}

caption_metric_map = {
    "Fitness": "fitness",
    "Precision": "precision",
    "F1-score": "F1-score",
    "Simplicity": "simplicity",
    "No. of Fallthroughs": "number of fallthroughs"
}

def get_config_display_name(self_loops, strict_end, parallel_end):
    """Returns the descriptive string name for the configuration."""
    if not self_loops and not strict_end and not parallel_end:
        return "No Heuristics"
    
    parts = []
    if self_loops: parts.append("self-loops")
    if strict_end: parts.append("strict-end")
    if parallel_end: parts.append("parallel-end")
    return " $\mid$ ".join(parts)

def get_csv_filename(self_loops, strict_end, parallel_end):
    """Generates the exact file name format matching the parameters."""
    return (
        f"IMf_results_translucent_self_loops_{self_loops}_"
        f"strict_end_activities_{strict_end}_"
        f"parallel_end_activities_heuristic_{parallel_end}_"
        f"remove_arcs_heuristics_False_add_arcs_heuristics_False.csv"
    )

def find_overlapping_configs(config_dfs, column, imf_df=None, imf_column=None):
    """Groups configurations that have identical values across all thresholds."""
    groups = []
    
    # Build the dictionary of dataframes to compare
    dfs_to_compare = {}
    for config_name, df in config_dfs.items():
        if column in df.columns:
            dfs_to_compare[config_name] = (df, column)
            
    # Include standard IMf values if provided for standard metric curves
    if imf_df is not None and imf_column in imf_df.columns:
        dfs_to_compare["IMf"] = (imf_df, imf_column)
        
    for config_name, (df, col) in dfs_to_compare.items():
        if col not in df.columns:
            continue
        
        # Ensure rows are perfectly sorted by threshold before comparing arrays
        df_sorted = df.sort_values("threshold")
        current_values = df_sorted[col].to_numpy()
        
        matched = False
        for group in groups:
            reference_config = group[0]
            ref_df, ref_col = dfs_to_compare[reference_config]
            ref_df_sorted = ref_df.sort_values("threshold")
            reference_values = ref_df_sorted[ref_col].to_numpy()
            
            # Check length and use floating point precision tolerance
            if len(current_values) == len(reference_values) and np.isclose(current_values, reference_values, atol=1e-9).all():
                group.append(config_name)
                matched = True
                break
                
        if not matched:
            groups.append([config_name])
            
    # Return only groups that actually contain overlapping elements (size > 1)
    return [g for g in groups if len(g) > 1]

def format_overlap_sentence(groups):
    """Converts the list of overlapping groups into clean English text."""
    if not groups:
        return "no overlapping plots"
    
    formatted_groups = [f"({', '.join(group)})" for group in groups]
    if len(formatted_groups) == 1:
        return f"identical plots for {formatted_groups[0]}"
    elif len(formatted_groups) == 2:
        return f"identical plots for {formatted_groups[0]} and {formatted_groups[1]}"
    else:
        return f"identical plots for {', '.join(formatted_groups[:-1])}, and {formatted_groups[-1]}"

# --- Main Generator ---
latex_output = []
param_combinations = list(itertools.product([False, True], repeat=3)) # 8 unique configurations

for ds_key, ds_name in datasets.items():
    for noise_key, noise_name in noise_types.items():
        
        # 1. Load all available configuration dataframes for this dataset + noise scenario
        config_dfs = {}
        for self_loops, strict_end, parallel_end in param_combinations:
            config_name = get_config_display_name(self_loops, strict_end, parallel_end)
            filename = get_csv_filename(self_loops, strict_end, parallel_end)
            file_path = os.path.join(CSV_BASE_DIR, ds_key, "results","translucent", noise_key, filename)
            
            if os.path.exists(file_path):
                try:
                    config_dfs[config_name] = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Warning: Could not read {file_path}. Error: {e}")
                    
        # Load standard baseline IMf dataframe for this setup
        imf_file_path = os.path.join(CSV_BASE_DIR, ds_key, "results", "IMf", noise_key, "IMf_results.csv")
        imf_df = None
        if os.path.exists(imf_file_path):
            try:
                imf_df = pd.read_csv(imf_file_path)
            except Exception as e:
                print(f"Warning: Could not read {imf_file_path}. Error: {e}")
        
        if not config_dfs:
            # Skip if no configurations were found for this setup
            continue

        # 2. Process each metric plot for this dataset + noise scenario
        for metric_name, variants in metrics_config.items():
            caption_parts = []
            
            # Check standard variant curves
            col_tf, col_ts = variants["standard"]
            imf_col = col_tf[:-3]  # Extracts base column name (e.g., 'fitness_tf' -> 'fitness')
            
            tf_groups = find_overlapping_configs(config_dfs, col_tf, imf_df, imf_col)
            ts_groups = find_overlapping_configs(config_dfs, col_ts, imf_df, imf_col)
            
            if "translucent" in variants:
                if tf_groups:
                    caption_parts.append(f"For normal values and the IMftf, there are {format_overlap_sentence(tf_groups)}.")
                if ts_groups:
                    caption_parts.append(f"For normal values and the IMfts, there are {format_overlap_sentence(ts_groups)}.")
            else:
                if tf_groups:
                    caption_parts.append(f"For the IMftf, there are {format_overlap_sentence(tf_groups)}.")
                if ts_groups:
                    caption_parts.append(f"For the IMfts, there are {format_overlap_sentence(ts_groups)}.")
            
            # Check translucent variant curves if applicable (IMf does not have translucent variant metrics)
            if "translucent" in variants:
                col_t_tf, col_t_ts = variants["translucent"]
                t_tf_groups = find_overlapping_configs(config_dfs, col_t_tf)
                t_ts_groups = find_overlapping_configs(config_dfs, col_t_ts)
                
                if t_tf_groups:
                    caption_parts.append(f"For translucent values and the IMftf, there are {format_overlap_sentence(t_tf_groups)}.")
                if t_ts_groups:
                    caption_parts.append(f"For translucent values and the IMfts, there are {format_overlap_sentence(t_ts_groups)}.")
                    
            # 3. Formulate the dynamic caption text
            overlap_caption_text = " ".join(caption_parts)
            
            # Construct figure path according to blueprint layout rules
            img_filename = f"{ds_name}_{noise_name}_{metric_name}.pdf"
            full_img_path = f"{FIGURES_BASE_PATH}/{img_filename}"
            
            # Sanitize labels safely for latex compilation keys
            clean_metric_label = metric_name.lower().replace('.', '').replace(' ', '_').replace('-', '_')
            label_name = f"fig:{ds_key}_{noise_key}_{clean_metric_label}"
            caption_metric_name = caption_metric_map[metric_name]
            
            # 4. Generate LaTeX code blocks
            latex_block = (
                f"\\begin{{figure}}[htbp]\n"
                f"    \\centering\n"
                f"    \\includegraphics[width=\\textwidth]{{{full_img_path}}}\n"
                f"    \\caption{{Evaluation of \\textbf{{{caption_metric_name}}} for the \\textbf{{{ds_name.lower()}}} dataset and the \\textbf{{{noise_name.lower()}}} noise type. {overlap_caption_text}}}\n"
                f"    \\label{{{label_name}}}\n"
                f"\\end{{figure}}\n"
            )
            
            latex_output.append(latex_block)

# Save code chunks to text file
with open(OUTPUT_TXT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_output))

print(f"Done! Generated {len(latex_output)} layout blocks inside '{OUTPUT_TXT_FILE}'.")