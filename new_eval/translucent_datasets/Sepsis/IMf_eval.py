# Conducts in parallel the evaluatioon of the IMf for the different noise types

from functools import partial
import multiprocessing as mp
import pandas as pd
import os
from pathlib import Path
import pm4py
import subprocess.CalledProcessError
import time
import copy

from new_eval.translucent_datasets.generate_noisy_datasets import get_noisy_log
from new_eval.utils.make_rooted import add_artificial_start_and_end_activities_translucent
from translucent_precision.main import translucent_precision_score
from evaluation.translucent_f_1_score import compute_f_1_scores
from translucent_fitness.fitness import calculate_log_fitness

LOG_NAME = "Sepsis"

def write_error_to_file(error, noise_type):
    file_path = Path(f"new_eval/translucent_datasets/{LOG_NAME}/results/{noise_type}/error_IMf.txt")
    with open(file_path, "w") as f:
        f.write(str(error.returncode) + "\n"
                + str(error.output) + "\n")

def evaluate_noise_type(noise_type, base_log):
    local_log = copy.deepcopy(base_log) 
    if not noise_type: # base log without noise
            noise_type = "base"
            log = pm4py.convert.convert_to_event_log(local_log)
    else: # Add noise to the log
        log = get_noisy_log(local_log, noise_type)
        log = pm4py.convert.convert_to_event_log(log)
        
    # Make the log rooted by adding artificial start and end events
    log = add_artificial_start_and_end_activities_translucent(log)
    
    filter_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    IMf_results = {}
    
    for f in filter_values:
        IMf_results[f] = {}
        
        try:
        # First excecute petrify and save the results, as these are needed for the eval graphs
            start = time.time()
            net, im, fm, max_memory = pm4py.discovery.discover_petri_net_inductive(log, noise_threshold=f)
            IMf_results[f]["time"] = time.time() - start
            IMf_results[f]["RAM"] = max_memory # in KB
            IMf_results[f]["RAM"] = max_memory / (1024) # Convert to MB
            
            # Calculate the scores
            IMf_results[f]["fitness"] = pm4py.conformance.fitness_alignments(log, net, im, fm)["log_fitness"]
            IMf_results[f]["precision"] = pm4py.conformance.precision_alignments(log, net, im, fm)
            IMf_results[f]["translucent_fitness"] = calculate_log_fitness(log, net, im, fm)
            IMf_results[f]["translucent_precision"] = translucent_precision_score(log, net, im, fm)
            IMf_results[f]["f_1_score"] = (2 * IMf_results[f]["fitness"] * IMf_results[f]["precision"]) / (IMf_results[f]["fitness"] + IMf_results[f]["precision"])
            IMf_results[f]["translucent_f_1_score"] = (2 * IMf_results[f]["translucent_fitness"] * IMf_results[f]["translucent_precision"]) / (IMf_results[f]["translucent_fitness"] + IMf_results[f]["translucent_precision"])
            IMf_results[f]["failed"] = False
            
        
        except Exception as e:
            if isinstance(e, subprocess.CalledProcessError): # Probably ran out of memory
                print(f"IMf failed for noise type {noise_type} with error: {e}")
                write_error_to_file(e, noise_type)
                IMf_results[f]["RAM"] = 0
                IMf_results[f]["time"] = 0
                IMf_results[f]["fitness"] = 0
                IMf_results[f]["precision"] = 0
                IMf_results[f]["translucent_fitness"] = 0
                IMf_results[f]["translucent_precision"] = 0
                IMf_results[f]["f_1_score"] = 0
                IMf_results[f]["translucent_f_1_score"] = 0
                IMf_results[f]["failed"] = True
                
                
            else: # Some other error occurred
                print(f"An unexpected error occurred for noise type {noise_type} with error: {e}")
                write_error_to_file(e, noise_type)
                raise e # Rethrow the error, as it is not expected and should be fixed
    
 
    # Create Folder for the noise type if it doesn't exist
    results_folder = Path(f"new_eval/translucent_datasets/{LOG_NAME}/results/{noise_type}")
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # Store the results as a .csv file
    results_path = Path(f"new_eval/translucent_datasets/{LOG_NAME}/results/{noise_type}/IMf_results.csv")
    results_df = pd.DataFrame([IMf_results])
    results_df.to_csv(results_path, index=False)
    
    


if __name__ == "__main__":
    
    noise_types = [False, "add_enabled", "remove_enabled", "add_events", "change_events"]
    
    path_to_log = Path(f"new_eval/translucent_datasets/{LOG_NAME}/{LOG_NAME}_0.2.csv")
    
    # Import the log
    base_log = pd.read_csv(path_to_log)
    
    func = partial(evaluate_noise_type, base_log=base_log)

    # 3. Use the pool
    num_processes = min(mp.cpu_count(), len(noise_types))
    
    with mp.Pool(processes=num_processes) as pool:
        # pool.map automatically passes each item in noise_types as the 
        # first argument to 'func', while base_log remains fixed.
        pool.map(func, noise_types)
    
    print( LOG_NAME + ": IMf evaluation completed for all noise types.")
    
    


