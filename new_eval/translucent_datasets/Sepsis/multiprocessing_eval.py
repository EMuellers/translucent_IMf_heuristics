import multiprocessing as mp
import pandas as pd
import os
from pathlib import Path
import pm4py
import subprocess.CalledProcessError
import time

from new_eval.translucent_datasets.generate_noisy_datasets import get_noisy_log
from new_eval.utils.discover_model_with_regions import discover_net_with_regions_from_rooted_log
from new_eval.utils.make_rooted import add_artificial_start_and_end_activities_translucent


"""
This script evaluates our approach with different parameters in parallel using python' multiprocessing library.
"""

LOG_NAME = "Sepsis"

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

def write_error_to_file(error, noise_type):
    file_path = Path(f"new_eval/translucent_datasets/{LOG_NAME}/results/{noise_type}/error.txt")
    with open(file_path, "w") as f:
        f.write(str(error.returncode) + "\n"
                + str(error.output) + "\n")
    

def conduct_evaluation(params):
    pass

if __name__ == "__main__":
    # Generate parameter configurations
    parameter_configs = generate_parameter_configs()
    
    noise_types = [False, "add_enabled", "remove_enabled", "add_events", "change_events"]
    
    path_to_log = Path(f"new_eval/translucent_datasets/{LOG_NAME}/{LOG_NAME}_0.2.csv")
    
    # Import the log
    base_log = pd.read_csv(path_to_log)
    
    # Evaluate log for every noise type
    for noise_type in noise_types:
        
        if not noise_type: # base log without noise
            noise_type = "base"
            log = pm4py.convert.convert_to_event_log(base_log)
        else: # Add noise to the log
            log = get_noisy_log(base_log, noise_type)
            log = pm4py.convert.convert_to_event_log(log)
            
        # Make the log rooted by adding artificial start and end events
        log = add_artificial_start_and_end_activities_translucent(log)
        
        petrify_results = {}
        
        try:
        # First excecute petrify and save the results, as these are needed for the eval graphs
            start = time.time()
            net, im, fm = discover_net_with_regions_from_rooted_log(log, log_name=LOG_NAME + "_" + noise_type)
            petrify_results["time"] = time.time() - start
        
        except Exception as e:
            if isinstance(e, subprocess.CalledProcessError): # Probably ran out of memory
                print(f"Petrify failed for noise type {noise_type} with error: {e}")
                write_error_to_file(e, noise_type)
                petrify_results["RAM"] = 0
                petrify_results["time"] = 0
                petrify_results["fitness"] = 0
                petrify_results["precision"] = 0
                petrify_results["translucent_fitness"] = 0
                petrify_results["translucent_precision"] = 0
                
                
            else: # Some other error occurred
                print(f"An unexpected error occurred for noise type {noise_type} with error: {e}")
                write_error_to_file(e, noise_type)
                raise e # Rethrow the error, as it is not expected and should be fixed
        
        
        # Parallelize the evaluation for different parameter settings
        
        
        
            
    
    

    # Create a pool of workers
    # Result files are saved by each worker process in the results folder
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Map the evaluation function to the parameter configurations
        pool.map(evaluate_model, parameter_configs)