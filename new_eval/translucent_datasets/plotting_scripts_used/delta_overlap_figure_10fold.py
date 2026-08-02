import os
import glob
import re
import pandas as pd
import numpy as np

# --- Configuration ---
# Adjust this path to match your local folder structure.
CSV_BASE_DIR = r"C:\eval_fall\new_eval\translucent_datasets" 
OUTPUT_TXT_FILE = r"C:\Users\elias\OneDrive - Students RWTH Aachen University\Masterarbeit\thesis-main\appendix_eval_figures\figures_delta_10fold.tex"
FIGURES_BASE_PATH = "./figures/eval/plots/delta_heur_10fold"

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

# Focused on Fitness to match your 10-fold delta visualization output
metrics_config = {
    "Fitness": {
        "standard": ("fitness_tf", "fitness_ts"),
        "translucent": ("translucent_fitness_tf", "translucent_fitness_ts")
    }
}

caption_metric_map = {
    "Fitness": "fitness"
}

# Your custom abbreviation dictionary map
shorten_map = {
    "confidence": "conf.",
    "dependency score": "dep.",
    "support": "sup.",
    "parallel relationship frequency": "par.",
    "exclusive choice frequency": "excl.",
}

NO_HEURISTICS_FILE_NAME = "IMf_10fold_translucent_self_loops_False_strict_end_activities_False_parallel_end_activities_heuristic_False_remove_arcs_heuristics_False_add_arcs_heuristics_False.csv"

def prettify_and_shorten_config_label(config_string: str) -> str:
    """Converts raw filename into shortened, publication-ready legend strings."""
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
    if len(parts) > 0 and '_' in parts[0]:
        first_key, first_value = parts[0].rsplit('_', 1)
        result[first_key] = first_value.lstrip('_')

    for i in range(1, len(parts), 2):
        key = parts[i]
        value = parts[i + 1].lstrip('_')
        result[key] = value

    # Keep only active (non-False) entries
    result = {k: v for k, v in result.items() if v != 'False'}
    if not result:
        return "No Heuristics"
    
    rename_map = {
        "remove_arcs_heuristics": "remove_arcs",
        "add_arcs_heuristics": "add_arcs",
    }
    result = {rename_map.get(k, k): v for k, v in result.items()}
    result = {k: v for k, v in result.items() if (v != 'False' and v != "True")}
    
    result_string = " $\mid$ ".join(result.values()).replace("_", " ")

    # Apply custom abbreviation mapping cleanly
    for long_name, short_name in shorten_map.items():
        result_string = result_string.replace(long_name, short_name)
        
    return result_string

def find_overlapping_configs(config_dfs, column, imf_df=None, imf_column=None):
    """Groups configurations that have identical values across all thresholds after averaging folds."""
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
        
        # Group by threshold and compute the mean across all 10 folds before comparing arrays
        df_avg = df.groupby("threshold")[col].mean().reset_index().sort_values("threshold")
        current_values = df_avg[col].to_numpy()
        
        matched = False
        for group in groups:
            reference_config = group[0]
            ref_df, ref_col = dfs_to_compare[reference_config]
            ref_df_avg = ref_df.groupby("threshold")[ref_col].mean().reset_index().sort_values("threshold")
            reference_values = ref_df_avg[ref_col].to_numpy()
            
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

for ds_key, ds_name in datasets.items():
    for noise_key, noise_name in noise_types.items():
        
        results_folder = os.path.join(CSV_BASE_DIR, ds_key, "results", "translucent_delta_10fold", noise_key)
        if not os.path.exists(results_folder):
            continue

        # 1. Dynamically read files present in the 10-fold delta folder
        config_dfs = {}
        search_path = os.path.join(results_folder, "IMf_10fold_translucent_*.csv")
        for file_path in glob.glob(search_path):
            filename = os.path.basename(file_path)
            config_key = filename.replace("IMf_10fold_", "").replace(".csv", "")
            config_name = prettify_and_shorten_config_label(config_key)
            
            try:
                config_dfs[config_name] = pd.read_csv(file_path)
            except Exception as e:
                print(f"Warning: Could not read {file_path}. Error: {e}")
                    
        # Load No Heuristics baseline from minor_10fold folder if missing
        no_heur_path = os.path.join(CSV_BASE_DIR, ds_key, "results", "translucent_minor_10fold", noise_key, NO_HEURISTICS_FILE_NAME)
        if os.path.exists(no_heur_path) and "No Heuristics" not in config_dfs:
            try:
                config_dfs["No Heuristics"] = pd.read_csv(no_heur_path)
            except Exception as e:
                print(f"Warning: Could not read {no_heur_path}. Error: {e}")

        # Load standard baseline 10-fold IMf dataframe
        imf_file_path = os.path.join(CSV_BASE_DIR, ds_key, "results", "IMf_10fold", noise_key, "IMf_results.csv")
        imf_df = None
        if os.path.exists(imf_file_path):
            try:
                imf_df = pd.read_csv(imf_file_path)
            except Exception as e:
                print(f"Warning: Could not read {imf_file_path}. Error: {e}")
        
        if not config_dfs:
            continue

        # 2. Process the metric plots
        for metric_name, variants in metrics_config.items():
            caption_parts = []
            
            # Check standard variant curves
            col_tf, col_ts = variants["standard"]
            imf_col = col_tf[:-3]  # Extracts base column name ('fitness_tf' -> 'fitness')
            
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
            
            # Check translucent variant curves
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
            img_filename = f"{ds_name}_{noise_name}_{metric_name}_10fold.pdf"
            full_img_path = f"{FIGURES_BASE_PATH}/{img_filename}"
            
            # Sanitize labels safely for latex compilation keys
            clean_metric_label = metric_name.lower().replace('.', '').replace(' ', '_').replace('-', '_')
            label_name = f"fig:{ds_key}_{noise_key}_{clean_metric_label}_10fold_delta"
            caption_metric_name = caption_metric_map[metric_name]
            
            # 4. Generate LaTeX code blocks
            latex_block = (
                f"\\begin{{figure}}[htbp]\n"
                f"    \\centering\n"
                f"    \\includegraphics[width=\\textwidth]{{{full_img_path}}}\n"
                f"    \\caption{{Evaluation of \\textbf{{{caption_metric_name}}} for the \\textbf{{{ds_name.lower()}}} dataset and the \\textbf{{{noise_name.lower()}}} noise type under 10-fold cross-validation. {overlap_caption_text}}}\n"
                f"    \\label{{{label_name}}}\n"
                f"\\end{{figure}}\n"
            )
            
            latex_output.append(latex_block)

# Save code chunks to text file
os.makedirs(os.path.dirname(OUTPUT_TXT_FILE), exist_ok=True)
with open(OUTPUT_TXT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_output))

print(f"Done! Generated {len(latex_output)} layout blocks inside '{OUTPUT_TXT_FILE}'.")