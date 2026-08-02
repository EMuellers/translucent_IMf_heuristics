import os
import glob
import re
import pandas as pd
import numpy as np

# --- Configuration ---
CSV_BASE_DIR = r"C:\eval_fall\new_eval\translucent_datasets" 
# Combined output target destination
OUTPUT_TXT_FILE = r"C:\Users\elias\OneDrive - Students RWTH Aachen University\Masterarbeit\thesis-main\appendix_eval_figures\combined_figures_delta.tex"

# Distinct path variables for graphic assets
FIGURES_BASE_PATH_ALL = "./figures/eval/plots/delta_heur"
FIGURES_BASE_PATH_10FOLD = "./figures/eval/plots/delta_heur_10fold"

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
    "F1 Score": {
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
    "F1 Score": "F1-score",
    "Simplicity": "simplicity",
    "No. of Fallthroughs": "number of fallthroughs"
}

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

    for long_name, short_name in shorten_map.items():
        result_string = result_string.replace(long_name, short_name)
        
    return result_string

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

for ds_key, ds_name in datasets.items():
    for noise_key, noise_name in noise_types.items():
        
        # ---------------------------------------------------------------------
        # PART 1: Standard Figures (All Metrics)
        # ---------------------------------------------------------------------
        results_folder_all = os.path.join(CSV_BASE_DIR, ds_key, "results", "translucent_delta_all", noise_key)
        if os.path.exists(results_folder_all):
            config_dfs_all = {}
            search_path_all = os.path.join(results_folder_all, "IMf_results_*.csv")
            for file_path in glob.glob(search_path_all):
                filename = os.path.basename(file_path)
                config_key = filename.replace("IMf_results_", "").replace(".csv", "")
                config_name = prettify_and_shorten_config_label(config_key)
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
                    label_name = f"fig:{ds_key}_{noise_key}_{clean_metric_label}_delta"
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
        results_folder_10fold = os.path.join(CSV_BASE_DIR, ds_key, "results", "translucent_delta_10fold", noise_key)
        if os.path.exists(results_folder_10fold):
            config_dfs_10fold = {}
            search_path_10fold = os.path.join(results_folder_10fold, "IMf_10fold_translucent_*.csv")
            for file_path in glob.glob(search_path_10fold):
                filename = os.path.basename(file_path)
                config_key = filename.replace("IMf_10fold_", "").replace(".csv", "")
                config_name = prettify_and_shorten_config_label(config_key)
                try:
                    config_dfs_10fold[config_name] = pd.read_csv(file_path)
                except Exception as e:
                    print(f"Warning: Could not read {file_path}. Error: {e}")
                        
            no_heur_path = os.path.join(CSV_BASE_DIR, ds_key, "results", "translucent_minor_10fold", noise_key, NO_HEURISTICS_FILE_NAME)
            if os.path.exists(no_heur_path) and "No Heuristics" not in config_dfs_10fold:
                try:
                    config_dfs_10fold["No Heuristics"] = pd.read_csv(no_heur_path)
                except Exception as e:
                    print(f"Warning: Could not read {no_heur_path}. Error: {e}")

            imf_file_path_10fold = os.path.join(CSV_BASE_DIR, ds_key, "results", "IMf_10fold", noise_key, "IMf_10fold_results.csv")
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
                    label_name = f"fig:{ds_key}_{noise_key}_{clean_metric_label}_10fold_delta"
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

# Save cumulative blocks into a single consolidated file
os.makedirs(os.path.dirname(OUTPUT_TXT_FILE), exist_ok=True)
with open(OUTPUT_TXT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(latex_output))

print(f"Done! Combined generation layout written directly into '{OUTPUT_TXT_FILE}'.")