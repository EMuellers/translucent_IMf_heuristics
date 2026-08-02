import warnings
from functools import partial
import multiprocessing as mp
import pandas as pd
import os
from pathlib import Path
import pm4py
from pm4py.util import constants
import time
import sys
if os.name != 'nt': # Windows
        sys.path.append("/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main")
else:
        sys.path.append("C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\TranslucentActivityRelationships-main")

from new_eval.utils.make_rooted import add_artificial_start_and_end_activities_translucent
from translucent_precision.main import translucent_precision_score_eval_version as translucent_precision_score
from translucent_fitness.fitness import calculate_log_fitness
from translucent_discovery.translucent_inductive_miner.translucent_base import discover_petri_net
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

# Sets Gurobi as the LP solver for pm4py 
constants.DEFAULT_LP_SOLVER = "gurobi"

LOG_NAME = "road_traffic_fine"

def generate_parameter_configs():
    parameter_list = []
    
    # Possible values for each parameter
    # These are excecuted for every translucent variant, so they do not need to be specified here
    
    translucent_self_loops = [True, False]
    strict_end_activities = [True, False]
    parallel_end_activities_heuristic = [True, False]
    remove_arcs_heuristic = [False, "dependency_score", "support", "confidence", "exclusive_choice_frequency"]
    add_arcs_heuristic = [False, "dependency_score", "support", "confidence", "parallel_relationship_frequency"]
    # Generate all combinations of parameters
    for self_loops in translucent_self_loops:
        for strict_end in strict_end_activities:
            for parallel_heuristic in parallel_end_activities_heuristic:
                for remove_heuristic in remove_arcs_heuristic:
                    for add_heuristic in add_arcs_heuristic:
                        # forbid some combinations that do not make sense
                        if remove_heuristic == add_heuristic and remove_heuristic != False:
                            continue 
                        parameter_list.append({
                            "translucent_self_loops": self_loops,
                            "strict_end_activities": strict_end,
                            "parallel_end_activities_heuristic": parallel_heuristic,
                            "remove_arcs_heuristic": remove_heuristic,
                            "add_arcs_heuristic": add_heuristic
                        })
    
    return parameter_list

def generate_minor_heuristics_parameter_configs():
    parameter_list = []
    
    # Possible values for each parameter
    # These are excecuted for every translucent variant, so they do not need to be specified here
    
    translucent_self_loops = [True, False]
    strict_end_activities = [True, False]
    parallel_end_activities_heuristic = [True, False]
    remove_arcs_heuristic = [False]
    add_arcs_heuristic = [False]
    # Generate all combinations of parameters
    for self_loops in translucent_self_loops:
        for strict_end in strict_end_activities:
            for parallel_heuristic in parallel_end_activities_heuristic:
                for remove_heuristic in remove_arcs_heuristic:
                    for add_heuristic in add_arcs_heuristic:
                        # forbid some combinations that do not make sense
                        if remove_heuristic == add_heuristic and remove_heuristic != False:
                            continue 
                        parameter_list.append({
                            "translucent_self_loops": self_loops,
                            "strict_end_activities": strict_end,
                            "parallel_end_activities_heuristic": parallel_heuristic,
                            "remove_arcs_heuristic": remove_heuristic,
                            "add_arcs_heuristic": add_heuristic
                        })
    
    return parameter_list

def write_error_to_file(error, noise_type, error_info=None):
    file_path = Path(f"new_eval/translucent_datasets/{LOG_NAME}/results/{noise_type}/error_IMf.txt")
    # Ensure directory exists before writing error
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w") as f:
        if error_info:
            f.write("\nFull traceback:\n" + error_info)
        f.write(str(error) + "\n")

# --- Worker Function for Parallel Filter Evaluation ---
def evaluate_single_filter(f, log, noise_type, parameters=None):
    """
    Evaluates a single filter value 'f' for a given 'log' and 'noise_type'.
    Returns a tuple: (f, result_dictionary)
    """
    result_data = {}
    
    #try:
    start = time.time()
    # Execute Inductive Miner f
    #TODO: Find a way to record RAM usage!
    max_memory = 0 # Placeholder for RAM usage
    net_tf, im_tf, fm_tf = discover_petri_net(log, {"translucent_variant": "IMtf", "tDFG_fall_through": True} | parameters, noise_threshold=f)
    duration_tf = time.time() - start
    start = time.time()
    net_ts, im_ts, fm_ts = discover_petri_net(log, {"translucent_variant": "IMts", "tDFG_fall_through": False} | parameters, noise_threshold=f)
    duration_ts = time.time() - start 
    
    # Calculate scores
    #print("Starting calculation of fitness for noise type: " + noise_type + " and filter: " + str(f) + " and parameters: " + str(parameters))
    #fitness_tf = pm4py.conformance.fitness_alignments(log, net_tf, im_tf, fm_tf)["log_fitness"]
    #fitness_ts = pm4py.conformance.fitness_alignments(log, net_ts, im_ts, fm_ts)["log_fitness"]
    print("Starting calculation of precision for noise type: " + noise_type + " and filter: " + str(f) + " and parameters: " + str(parameters))
    precision_tf = pm4py.conformance.precision_alignments(log, net_tf, im_tf, fm_tf)
    precision_ts = pm4py.conformance.precision_alignments(log, net_ts, im_ts, fm_ts)
    print("Starting calculation of translucent fitness and precision for IMtf for noise type: " + noise_type + " and filter: " + str(f))
    translucent_fitness_tf = calculate_log_fitness(log, net_tf, im_tf, fm_tf)
    translucent_precision_tf, fitness_tf = translucent_precision_score(log, net_tf, im_tf, fm_tf)
    print("Starting calculation of translucent fitness and precision for IMts for noise type: " + noise_type + " and filter: " + str(f) + " and parameters: " + str(parameters))
    translucent_fitness_ts = calculate_log_fitness(log, net_ts, im_ts, fm_ts)
    translucent_precision_ts, fitness_ts = translucent_precision_score(log, net_ts, im_ts, fm_ts)
    
    # Calculate F1 scores (handle division by zero if needed, though standard formula assumes non-zero sum usually)
    denom_f1_tf = fitness_tf + precision_tf
    f_1_score_tf = (2 * fitness_tf * precision_tf) / denom_f1_tf if denom_f1_tf > 0 else 0
    
    denom_f1_ts = fitness_ts + precision_ts
    f_1_score_ts = (2 * fitness_ts * precision_ts) / denom_f1_ts if denom_f1_ts > 0 else 0
    
    # Calculate F1 scores for translucent metrics
    denom_tf1_tf = translucent_fitness_tf + translucent_precision_tf
    translucent_f_1_score_tf = (2 * translucent_fitness_tf * translucent_precision_tf) / denom_tf1_tf if denom_tf1_tf > 0 else 0
    
    denom_tf1_ts = translucent_fitness_ts + translucent_precision_ts
    translucent_f_1_score_ts = (2 * translucent_fitness_ts * translucent_precision_ts) / denom_tf1_ts if denom_tf1_ts > 0 else 0
    
    simplicity_tf = len(net_tf.places) + len(net_tf.transitions) + len(net_tf.arcs)
    simplicity_ts = len(net_ts.places) + len(net_ts.transitions) + len(net_ts.arcs)
    
    result_data = {
        "time_tf": duration_tf,
        "time_ts": duration_ts,
        "RAM": max_memory / 1024, # Convert to MB
        "fitness_tf": fitness_tf,
        "precision_tf": precision_tf,
        "f_1_score_tf": f_1_score_tf,
        "fitness_ts": fitness_ts,
        "precision_ts": precision_ts,
        "f_1_score_ts": f_1_score_ts,
        "translucent_fitness_tf": translucent_fitness_tf,
        "translucent_precision_tf": translucent_precision_tf,
        "translucent_fitness_ts": translucent_fitness_ts,
        "translucent_precision_ts": translucent_precision_ts,
        "translucent_f_1_score_tf": translucent_f_1_score_tf,
        "translucent_f_1_score_ts": translucent_f_1_score_ts,
        "simplicity_tf": simplicity_tf,
        "simplicity_ts": simplicity_ts,
        "failed": False
        }
   
    print(f"Completed evaluation for noise type: {noise_type}, filter: {f}, parameters: {parameters}. Result: {result_data}")    
    return f, result_data

# --- Worker Function for Parallel Filter Evaluation ---
def evaluate_single_filter_test(f, log, noise_type, parameters=None):
    """
    Evaluates a single filter value 'f' for a given 'log' and 'noise_type'.
    Returns a tuple: (f, result_dictionary)
    """
    result_data = {}
    
    #try:
    start = time.time()
    # Execute Inductive Miner f
    #TODO: Find a way to record RAM usage!
    max_memory = 0 # Placeholder for RAM usage
    net_tf, im_tf, fm_tf = discover_petri_net(log, {"translucent_variant": "IMtf", "tDFG_fall_through": True} | parameters, noise_threshold=f)
    duration_tf = time.time() - start
    start = time.time()
    net_ts, im_ts, fm_ts = discover_petri_net(log, {"translucent_variant": "IMts", "tDFG_fall_through": False} | parameters, noise_threshold=f)
    duration_ts = time.time() - start 
    
    print("Starting calculation of translucent fitness and precision for IMtf for noise type: " + noise_type + " and filter: " + str(f))
    translucent_fitness_tf = calculate_log_fitness(log, net_tf, im_tf, fm_tf)
    print("Starting calculation of translucent fitness and precision for IMts for noise type: " + noise_type + " and filter: " + str(f) + " and parameters: " + str(parameters))
    translucent_fitness_ts = calculate_log_fitness(log, net_ts, im_ts, fm_ts)
    
    
    
    simplicity_tf = len(net_tf.places) + len(net_tf.transitions) + len(net_tf.arcs)
    simplicity_ts = len(net_ts.places) + len(net_ts.transitions) + len(net_ts.arcs)
    
    result_data = {
        "time_tf": duration_tf,
        "time_ts": duration_ts,
        "RAM": max_memory / 1024, # Convert to MB
        "translucent_fitness_tf": translucent_fitness_tf,
        "translucent_fitness_ts": translucent_fitness_ts,
        "simplicity_tf": simplicity_tf,
        "simplicity_ts": simplicity_ts,
        "failed": False
        }
   
    print(f"Completed evaluation for noise type: {noise_type}, filter: {f}, parameters: {parameters}. Result: {result_data}")    
    return f, result_data

def prettify_config_for_file_name(config):
    # Convert a config dictionary to a string that is suitable for use in file names
    return "_".join([f"{k}_{v}" for k, v in config.items()])

def evaluate_noise_type(noise_type, log_path):
    
    # Handle noise type string/boolean logic
    noise_label = noise_type if noise_type else "base"
    
    if not noise_type: # base log without noise
        log_path = log_path / f"{LOG_NAME}_0.2.csv"
    else: # Add noise to the log
        log_path = log_path / f"{LOG_NAME}_0.2_{noise_type}.csv"
    
    log = pd.read_csv(log_path)
    log = pm4py.format_dataframe(log, case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp", timest_format="%Y-%m-%d %H:%M:%S%z")
    log = pm4py.convert_to_event_log(log)
        
    # Make the log rooted
    log = add_artificial_start_and_end_activities_translucent(log)
    
    filter_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    
    filter_values = [0.0,0.1, 0.2, 0.3, 0.4] #TODO!
    
    parameter_configs = generate_minor_heuristics_parameter_configs() # TODO: Change this for full eval!
    
    parameter_configs = parameter_configs[:2] # For testing only #TODO!
    
    for config in parameter_configs:
    
        # --- Parallel Execution over Filter Values ---
        print(f"Starting evaluation for noise type: {noise_label} and parameters: {config}")
        
        #evaluate_single_filter(f=0.0, log=log, noise_type=noise_label, parameters=config)#TODO!
        
        # Partial function binds the log and noise_type, leaving 'f' as the variable
        worker_func = partial(evaluate_single_filter_test, log=log, noise_type=noise_label, parameters=config)
        
        # Use all available CPUs for the filter values
        num_processes = min(mp.cpu_count(), len(filter_values))
        num_processes = 2 #TODO!
        
        with mp.Pool(processes=num_processes) as pool:
            # Returns a list of (f, result_dict) tuples
            results_list = pool.map(worker_func, filter_values)
    
        # Reconstruct the dictionary structure: {0: {...}, 0.1: {...}}
        IMf_results = dict(results_list)

        # Create Folder for the noise type if it doesn't exist
        results_folder = Path(f"new_eval/translucent_datasets/{LOG_NAME}/results/{noise_label}/translucent")
        results_folder.mkdir(parents=True, exist_ok=True)
    
        # Store the results as a .csv file, include the parameter settings in the file name
        results_path = results_folder / f"IMf_results_{prettify_config_for_file_name(config)}.csv"
        results_df = pd.DataFrame([IMf_results])
        results_df = pd.DataFrame.from_dict(IMf_results, orient='index')
        results_df.to_csv(results_path, index=True, index_label='threshold')
    
    print(f"Completed all parameter configs for noise type {noise_label}")


if __name__ == "__main__":
    
    noise_types = [False, "add_enabled", "remove_enabled", "add_events", "change_events"]
    noise_types = ["add_enabled"] #TODO!
    
    path_to_log = Path(f"new_eval/translucent_datasets/{LOG_NAME}")
    
    if os.name == 'nt': # Windows
        path_to_log = Path(f"C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\TranslucentActivityRelationships-main\\new_eval\\translucent_datasets\\{LOG_NAME}")
    
    # Sequential execution of noise types
    for noise in noise_types:
        evaluate_noise_type(noise, path_to_log)
    
    print(LOG_NAME + ": IMf_translucent evaluation completed for all noise types.")