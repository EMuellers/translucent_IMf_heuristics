import warnings
import traceback
from functools import partial
import multiprocessing as mp
import pandas as pd
import os
from pathlib import Path
import pm4py
import subprocess
import time
import copy
import sys
if os.name != 'nt': # Windows
        sys.path.append("/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main")

from new_eval.translucent_datasets.generate_noisy_datasets import get_noisy_log
from new_eval.utils.make_rooted import add_artificial_start_and_end_activities_translucent
from translucent_precision.main import translucent_precision_score
from evaluation.translucent_f_1_score import compute_f_1_scores
from translucent_fitness.fitness import calculate_log_fitness
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

LOG_NAME = "road_traffic_fine"

def write_error_to_file(error, noise_type, error_info=None):
    file_path = Path(f"new_eval/translucent_datasets/{LOG_NAME}/results/{noise_type}/error_IMf.txt")
    # Ensure directory exists before writing error
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w") as f:
        if error_info:
            f.write("\nFull traceback:\n" + error_info)
        f.write(str(error) + "\n")

# --- Worker Function for Parallel Filter Evaluation ---
def evaluate_single_filter(f, log, noise_type):
    """
    Evaluates a single filter value 'f' for a given 'log' and 'noise_type'.
    Returns a tuple: (f, result_dictionary)
    """
    result_data = {}
    
    try:
        start = time.time()
        # Execute Inductive Miner f
        #TODO: Find a way to record RAM usage!
        max_memory = 0 # Placeholder for RAM usage, as measuring it in Python is non-trivial and often requires external libraries or tools.
        net, im, fm = pm4py.discovery.discover_petri_net_inductive(log, noise_threshold=f)
        
        
        duration = time.time() - start
        
        # Calculate scores
        print("Starting calculation of fitness for noise type: " + noise_type + " and filter: " + str(f))
        fitness = pm4py.conformance.fitness_alignments(log, net, im, fm)["log_fitness"]
        print("Starting calculation of precision for noise type: " + noise_type + " and filter: " + str(f))
        precision = pm4py.conformance.precision_alignments(log, net, im, fm)
        print("Starting calculation of translucent fitness for noise type: " + noise_type + " and filter: " + str(f))
        translucent_fitness = calculate_log_fitness(log, net, im, fm)
        print("Starting calculation of translucent precision for noise type: " + noise_type + " and filter: " + str(f))
        translucent_precision = translucent_precision_score(log, net, im, fm)
        
        # Calculate F1 scores (handle division by zero if needed, though standard formula assumes non-zero sum usually)
        denom_f1 = fitness + precision
        f_1_score = (2 * fitness * precision) / denom_f1 if denom_f1 > 0 else 0
        
        denom_tf1 = translucent_fitness + translucent_precision
        translucent_f_1_score = (2 * translucent_fitness * translucent_precision) / denom_tf1 if denom_tf1 > 0 else 0
        
        simplicity = len(net.places) + len(net.transitions) + len(net.arcs)
        
        result_data = {
            "time": duration,
            "RAM": max_memory / 1024, # Convert to MB
            "fitness": fitness,
            "precision": precision,
            "translucent_fitness": translucent_fitness,
            "translucent_precision": translucent_precision,
            "f_1_score": f_1_score,
            "translucent_f_1_score": translucent_f_1_score,
            "simplicity": simplicity,
            "failed": False
        }

    except Exception as e:
        # Define empty result structure for failure
        failure_result = {
            "time": 0, "RAM": 0, "fitness": 0, "precision": 0,
            "translucent_fitness": 0, "translucent_precision": 0,
            "f_1_score": 0, "translucent_f_1_score": 0,
            "failed": True
        }
        error_info = traceback.format_exc()
        if isinstance(e, subprocess.CalledProcessError):
            print(f"IMf failed for noise type {noise_type}, filter {f} with error: {e}")
            write_error_to_file(e, noise_type, error_info)
            result_data = failure_result
        else:
            print(f"An unexpected error occurred for noise type {noise_type}, filter {f} with error: {e}")
            write_error_to_file(e, noise_type, error_info)
            # In parallel execution, raising an error stops the whole pool. 
            #raise e
        
    print(f"Completed evaluation for noise type: {noise_type}, filter: {f}. Result: {result_data}")    
    return f, result_data

def evaluate_noise_type(noise_type, base_log):
    local_log = copy.deepcopy(base_log) 
    
    # Handle noise type string/boolean logic
    noise_label = noise_type if noise_type else "base"
    
    if not noise_type: # base log without noise
        #local_log['time:timestamp'] = pd.to_datetime(local_log['time:timestamp'])
        # Create a list of columns to convert (everything except 'time')
        #cols_to_convert = local_log.columns.difference(['time:timestamp'])

        # Convert those columns to string
        #local_log[cols_to_convert] = local_log[cols_to_convert].astype(str)
        local_log = pm4py.format_dataframe(local_log, case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp", timest_format="%Y-%m-%d %H:%M:%S%z")
        log = pm4py.convert_to_event_log(local_log)
    else: # Add noise to the log
        log = get_noisy_log(local_log, noise_type)
        #log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
        # Create a list of columns to convert (everything except 'time')
        #cols_to_convert = log.columns.difference(['time:timestamp'])
        # Convert those columns to string
        #log[cols_to_convert] = log[cols_to_convert].astype(str)
        log = pm4py.format_dataframe(log, case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp", timest_format="%Y-%m-%d %H:%M:%S%z")
        log = pm4py.convert_to_event_log(log)
        
    # Make the log rooted
    log = add_artificial_start_and_end_activities_translucent(log)
    
    filter_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # --- Parallel Execution over Filter Values ---
    print(f"Starting evaluation for noise type: {noise_label}...")
    
    # Partial function binds the log and noise_type, leaving 'f' as the variable
    worker_func = partial(evaluate_single_filter, log=log, noise_type=noise_label)
    
    # Use all available CPUs for the filter values
    num_processes = min(mp.cpu_count(), len(filter_values))
    
    with mp.Pool(processes=num_processes) as pool:
        # Returns a list of (f, result_dict) tuples
        results_list = pool.map(worker_func, filter_values)
    
    # Reconstruct the dictionary structure: {0: {...}, 0.1: {...}}
    IMf_results = dict(results_list)

    # Create Folder for the noise type if it doesn't exist
    results_folder = Path(f"new_eval/translucent_datasets/{LOG_NAME}/results/{noise_label}")
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # Store the results as a .csv file
    results_path = results_folder / "IMf_results.csv"
    results_df = pd.DataFrame([IMf_results])
    results_df = pd.DataFrame.from_dict(IMf_results, orient='index')
    results_df.to_csv(results_path, index=True, index_label='threshold')
    
    print(f"Completed {noise_label}")


if __name__ == "__main__":
    
    noise_types = [False, "add_enabled", "remove_enabled", "add_events", "change_events"]
    
    path_to_log = Path(f"new_eval/translucent_datasets/{LOG_NAME}/{LOG_NAME}_0.2.csv")
    
    # Import the log
    base_log = pd.read_csv(path_to_log)
    
    # Sequential execution of noise types
    for noise in noise_types:
        evaluate_noise_type(noise, base_log)
    
    print(LOG_NAME + ": IMf evaluation completed for all noise types.")