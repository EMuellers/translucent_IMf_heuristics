import os
import itertools
import pandas as pd
import numpy as np

# --- Configuration ---
CSV_BASE_DIR = r"C:\eval_fall\new_eval\translucent_datasets" 
# Combined output target destination
OUTPUT_TXT_FILE = r"C:\Users\elias\OneDrive - Students RWTH Aachen University\Masterarbeit\thesis-main\appendix_eval_figures\combined_figures_single.tex"

# Distinct path variables for graphic assets
FIGURES_BASE_PATH_ALL = "./figures/eval/plots/minor_parameters"
FIGURES_BASE_PATH_10FOLD = "./figures/eval/plots/minor_parameters_10fold"

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

# Config structures kept separated to map correct columns/variants independently
metrics_config_all = {
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

metrics_config_10fold = {
    "Fitness": {
        "standard": ("fitness_tf", "fitness_ts"),
        "translucent": ("translucent_fitness_tf", "translucent_fitness_ts")
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

def get_csv_filename_all(self_loops, strict_end, parallel_end):
    """Generates the exact file name format matching the standard parameters."""
    return (
        f"IMf_results_translucent_self_loops_{self_loops}_"
        f"strict_end_activities_{strict_end}_"
        f"parallel_end_activities_heuristic_{parallel_end}_"
        f"remove_arcs_heuristics_False_add_arcs_heuristics_False.csv"
    )

def get_csv_filename_10fold(self_loops, strict_end, parallel_end):
    """Generates the exact file name format matching the 10fold parameters."""
    return (
        f"IMf_10fold_translucent_self_loops_{self_loops}_"
        f"strict_end_activities_{strict_end}_"
        f"parallel_end_activities_heuristic_{parallel_end}_"
        f"remove_arcs_heuristics_False_add_arcs_heuristics_False.csv"
    )

def find_overlapping_configs_all(config_dfs, column, imf_df=None, imf_column=None):
    """Groups configurations that have identical values across all thresholds."""
    groups = []
    dfs_to_compare = {}
    for config_name, df in config_dfs.items():
        if column in df.columns:
            dfs_to_compare[config_name] = (df, column)
            
    if imf_df is not None and imf_column in imf_df.columns:
        dfs_to_compare["IMf"] = (imf_df, imf_column)
        
    for config_name, (df, col) in dfs_to_compare.items():
        if col not in df.columns:
            continue
        
        df_sorted = df.sort_values("threshold")
        current_values = df_sorted[col].to_numpy()
        
        matched = False
        for group in groups:
            reference_config = group[0]
            ref_df, ref_col = dfs_to_compare[reference_config]
            ref_df_sorted = ref_df.sort_values("threshold")
            reference_values = ref_df_sorted[ref_col].to_numpy()
            
            if len(current_values) == len(reference_values) and np.isclose(current_values, reference_values, atol=1e-9).all():
                group.append(config_name)
                matched = True
                break
                
        if not matched:
            groups.append([config_name])
            
    return [g for g in groups if len(g) > 1]

def find_overlapping_configs_10fold(config_dfs, column, imf_df=None, imf_column=None):
    """Groups configurations that have identical averaged fold metrics across thresholds."""
    groups = []
    dfs_to_compare = {}
    for config_name, df in config_dfs.items():
        if column in df.columns:
            dfs_to_compare[config_name] = (df, column)
            
    if imf_df is not None and imf_column in imf_df.columns:
        dfs_to_compare["IMf"] = (imf_df, imf_column)
        
    for config_name, (df, col) in dfs_to_compare.items():
        if col not in df.columns:
            continue
        
        df_avg = df.groupby("threshold")[col].mean().reset_index().sort_values("threshold")
        current_values = df_avg[col].to_numpy()
        
        matched = False
        for group in groups:
            reference_config = group[0]
            ref_df, ref_col = dfs_to_compare[reference_config]
            ref_df_avg = ref_df.groupby("threshold")[ref_col].mean().reset_index().sort_values("threshold")
            reference_values = ref_df_avg[ref_col].to_numpy()
            
            if len(current_values) == len(reference_values) and np.isclose(current_values, reference_values, atol=1e-9).all():
                group.append(config_name)
                matched = True
                break
                
        if not matched:
            groups.append([config_name])
            
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

# --- Main Combined Generator Loop ---
latex_output = []
param_combinations = list(itertools.product([False, True], repeat=3)) # 8 unique configurations

for ds_key, ds_name in datasets.items():
    for noise_key, noise_name in noise_types.items():
        
        # ---------------------------------------------------------------------
        # PART 1: Standard Figures (All Metrics)
        # ---------------------------------------------------------------------
        config_dfs_all = {}
        for self_loops, strict_end, parallel_end in param_combinations:
            config_name = get_config_display_name(self_loops, strict_end, parallel_end)
            filename = get_csv_filename_all(self_loops, strict_end, parallel_end)
            file_path = os.path.join(CSV_BASE_DIR, ds_key, "results", "translucent", noise_key, filename)
            
            if os.path.exists(file_path):
                try:
                    config_dfs_all[config_name] = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Warning: Could not read {file_path}. Error: {e}")
                    
        imf_file_path_all = os.path.join(CSV_BASE_DIR, ds_key, "results", "IMf", noise_key, "IMf_results.csv")
        imf_df_all = None
        if os.path.exists(imf_file_path_all):
            try:
                imf_df_all = pd.read_csv(imf_file_path_all)
            except Exception as e:
                print(f"Warning: Could not read {imf_file_path_all}. Error: {e}")
        
        if config_dfs_all:
            for metric_name, variants in metrics_config_all.items():
                caption_parts = []
                col_tf, col_ts = variants["standard"]
                imf_col = col_tf[:-3]
                
                tf_groups = find_overlapping_configs_all(config_dfs_all, col_tf, imf_df_all, imf_col)
                ts_groups = find_overlapping_configs_all(config_dfs_all, col_ts, imf_df_all, imf_col)
                
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
                
                if "translucent" in variants:
                    col_t_tf, col_t_ts = variants["translucent"]
                    t_tf_groups = find_overlapping_configs_all(config_dfs_all, col_t_tf)
                    t_ts_groups = find_overlapping_configs_all(config_dfs_all, col_t_ts)
                    if t_tf_groups:
                        caption_parts.append(f"For translucent values and the IMftf, there are {format_overlap_sentence(t_tf_groups)}.")
                    if t_ts_groups:
                        caption_parts.append(f"For translucent values and the IMfts, there are {format_overlap_sentence(t_ts_groups)}.")
                        
                overlap_caption_text = " ".join(caption_parts)
                img_filename = f"{ds_name}_{noise_name}_{metric_name}.pdf"
                full_img_path = f"{FIGURES_BASE_PATH_ALL}/{img_filename}"
                
                clean_metric_label = metric_name.lower().replace('.', '').replace(' ', '_').replace('-', '_')
                label_name = f"fig:{ds_key}_{noise_key}_{clean_metric_label}"
                caption_metric_name = caption_metric_map[metric_name]
                
                latex_block = (
                    f"\\begin{{figure}}[htbp]\n"
                    f"    \\centering\n"
                    f"    \\includegraphics[width=\\textwidth]{{{full_img_path}}}\n"
                    f"    \\caption{{Evaluation of \\textbf{{{caption_metric_name}}} for the \\textbf{{{ds_name.lower()}}} dataset and the \\textbf{{{noise_name.lower()}}} noise type. {overlap_caption_text}}}\n"
                    f"    \\label{{{label_name}}}\n"
                    f"\\end{{figure}}\n"
                )
                latex_output.append(latex_block)

        # ---------------------------------------------------------------------
        # PART 2: Generalization Plot (10-Fold Cross-Validation Fitness)
        # ---------------------------------------------------------------------
        config_dfs_10fold = {}
        for self_loops, strict_end, parallel_end in param_combinations:
            config_name = get_config_display_name(self_loops, strict_end, parallel_end)
            filename = get_csv_filename_10fold(self_loops, strict_end, parallel_end)
            file_path = os.path.join(CSV_BASE_DIR, ds_key, "results", "translucent_minor_10fold", noise_key, filename)
            
            if os.path.exists(file_path):
                try:
                    config_dfs_10fold[config_name] = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Warning: Could not read {file_path}. Error: {e}")
                    
        imf_file_path_10fold = os.path.join(CSV_BASE_DIR, ds_key, "results", "IMf_10fold", noise_key, "IMf_results.csv")
        imf_df_10fold = None
        if os.path.exists(imf_file_path_10fold):
            try:
                imf_df_10fold = pd.read_csv(imf_file_path_10fold)
            except Exception as e:
                print(f"Warning: Could not read {imf_file_path_10fold}. Error: {e}")
        
        if config_dfs_10fold:
            for metric_name, variants in metrics_config_10fold.items():
                caption_parts = []
                col_tf, col_ts = variants["standard"]
                imf_col = col_tf[:-3]
                
                tf_groups = find_overlapping_configs_10fold(config_dfs_10fold, col_tf, imf_df_10fold, imf_col)
                ts_groups = find_overlapping_configs_10fold(config_dfs_10fold, col_ts, imf_df_10fold, imf_col)
                
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
                
                if "translucent" in variants:
                    col_t_tf, col_t_ts = variants["translucent"]
                    t_tf_groups = find_overlapping_configs_10fold(config_dfs_10fold, col_t_tf)
                    t_ts_groups = find_overlapping_configs_10fold(config_dfs_10fold, col_t_ts)
                    if t_tf_groups:
                        caption_parts.append(f"For translucent values and the IMftf, there are {format_overlap_sentence(t_tf_groups)}.")
                    if t_ts_groups:
                        caption_parts.append(f"For translucent values and the IMfts, there are {format_overlap_sentence(t_ts_groups)}.")
                        
                overlap_caption_text = " ".join(caption_parts)
                img_filename = f"{ds_name}_{noise_name}_{metric_name}_10fold.pdf"
                full_img_path = f"{FIGURES_BASE_PATH_10FOLD}/{img_filename}"
                
                clean_metric_label = metric_name.lower().replace('.', '').replace(' ', '_').replace('-', '_')
                label_name = f"fig:{ds_key}_{noise_key}_{clean_metric_label}_10fold"
                caption_metric_name = caption_metric_map[metric_name]
                
                latex_block = (
                    f"\\begin{{figure}}[htbp]\n"
                    f"    \\centering\n"
                    f"    \\includegraphics[width=\\textwidth]{{{full_img_path}}}\n"
                    f"    \\caption{{Evaluation of \\textbf{{generalization}} for the \\textbf{{{ds_name.lower()}}} dataset and the \\textbf{{{noise_name.lower()}}} noise type using fitness and 10-fold cross-validation. {overlap_caption_text}}}\n"
                    f"    \\label{{{label_name}}}\n"
                    f"\\end{{figure}}\n"
                )
                latex_output.append(latex_block)

# Save the unified block structure into a single file
os.makedirs(os.path.dirname(OUTPUT_TXT_FILE), exist_ok=True)
with open(OUTPUT_TXT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_output))

print(f"Done! Combined generation layout written directly into '{OUTPUT_TXT_FILE}'.")