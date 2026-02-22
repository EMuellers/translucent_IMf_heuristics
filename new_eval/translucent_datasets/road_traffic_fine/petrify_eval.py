#Conducts in parallel the evaluation of the petrify method for the different noise types
import warnings
import traceback
from functools import partial
import multiprocessing as mp
import pandas as pd
import os
from pathlib import Path
import pm4py
from subprocess import CalledProcessError
import time
import copy
import sys
if os.name != 'nt': # Windows
        sys.path.append("/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main")
else:
        sys.path.append("C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\TranslucentActivityRelationships-main")

from new_eval.translucent_datasets.generate_noisy_datasets import get_noisy_log
from new_eval.utils.discover_model_with_regions import discover_net_with_regions_from_rooted_log
from new_eval.utils.make_rooted import add_artificial_start_and_end_activities_translucent
from translucent_precision.main import translucent_precision_score
from evaluation.translucent_f_1_score import compute_f_1_scores
from translucent_fitness.fitness import calculate_log_fitness
from translucent_discovery.translucent_inductive_miner.translucent_datatype import translucent_log_to_tcl

from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

LOG_NAME = "road_traffic_fine"

def write_error_to_file(error, noise_type, error_info=None):
    file_path = Path(f"new_eval/translucent_datasets/{LOG_NAME}/results/{noise_type}/error_petrify.txt")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(str(error) + "\n")
        if error_info:
            f.write("\nFull traceback:\n" + error_info)

def evaluate_noise_type(noise_type, base_log):
    local_log = copy.deepcopy(base_log) 
    if not noise_type: # base log without noise
        noise_type = "base"
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
        
    # Make the log rooted by adding artificial start and end events
    log = add_artificial_start_and_end_activities_translucent(log)
    
    tcl_log = translucent_log_to_tcl(log)
    
    petrify_results = {}
        
    try:
    # First excecute petrify and save the results, as these are needed for the eval graphs
        start = time.time()
        if os.name == 'nt': # Windows
            net, im, fm = discover_net_with_regions_from_rooted_log(tcl_log, log_name=LOG_NAME + "_" + noise_type)
            max_memory = 0 # Placeholder for RAM usage
        else:
            net, im, fm, max_memory = discover_net_with_regions_from_rooted_log(tcl_log, log_name=LOG_NAME + "_" + noise_type)
        petrify_results["time"] = time.time() - start
        petrify_results["RAM"] = max_memory # in KB
        petrify_results["RAM"] = max_memory / (1024) # Convert to MB
        
        # Calculate the scores
        print("Starting calculation of fitness for noise type " + noise_type)
        petrify_results["fitness"] = pm4py.conformance.fitness_alignments(log, net, im, fm)["log_fitness"]
        print("Starting calculation of precision for noise type " + noise_type)
        petrify_results["precision"] = pm4py.conformance.precision_alignments(log, net, im, fm)
        print("Starting calculation of translucent fitness for noise type " + noise_type)
        petrify_results["translucent_fitness"] = calculate_log_fitness(log, net, im, fm)
        print("Starting calculation of translucent precision for noise type " + noise_type)
        petrify_results["translucent_precision"] = translucent_precision_score(log, net, im, fm)
        petrify_results["f_1_score"] = (2 * petrify_results["fitness"] * petrify_results["precision"]) / (petrify_results["fitness"] + petrify_results["precision"]) if (petrify_results["fitness"] + petrify_results["precision"]) > 0 else 0
        petrify_results["translucent_f_1_score"] = (2 * petrify_results["translucent_fitness"] * petrify_results["translucent_precision"]) / (petrify_results["translucent_fitness"] + petrify_results["translucent_precision"]) if (petrify_results["translucent_fitness"] + petrify_results["translucent_precision"]) > 0 else 0
        petrify_results["simplicity"] = len(net.places) + len(net.transitions) + len(net.arcs)
        petrify_results["failed"] = False
        
    
    except Exception as e:
        if isinstance(e, CalledProcessError): # Probably ran out of memory
            print(f"Petrify failed for noise type {noise_type} with error: {e}")
            error_info = traceback.format_exc()
            write_error_to_file(e, noise_type, error_info)
            petrify_results["RAM"] = 0
            petrify_results["time"] = 0
            petrify_results["fitness"] = 0
            petrify_results["precision"] = 0
            petrify_results["translucent_fitness"] = 0
            petrify_results["translucent_precision"] = 0
            petrify_results["f_1_score"] = 0
            petrify_results["translucent_f_1_score"] = 0
            petrify_results["simplicity"] = 0
            petrify_results["failed"] = True
            
            
        else: # Some other error occurred
            print(f"An unexpected error occurred for noise type {noise_type} with error: {e}")
            error_info = traceback.format_exc()
            write_error_to_file(e, noise_type, error_info)
            #raise e # Rethrow the error, as it is not expected and should be fixed
    
 
    # Create Folder for the noise type if it doesn't exist
    results_folder = Path(f"new_eval/translucent_datasets/{LOG_NAME}/results/{noise_type}")
    results_folder.mkdir(parents=True, exist_ok=True)
    
    # Store the results as a .csv file
    results_path = Path(f"new_eval/translucent_datasets/{LOG_NAME}/results/{noise_type}/petrify_results.csv")
    results_df = pd.DataFrame([petrify_results])
    results_df.to_csv(results_path, index=False)
    print(f"Petrify evaluation completed for noise type {noise_type}. Results stored at {results_path}")
    
    


if __name__ == "__main__":
    
    noise_types = [False, "add_enabled", "remove_enabled", "add_events", "change_events"]
    
    path_to_log = Path(f"new_eval/translucent_datasets/{LOG_NAME}/{LOG_NAME}_0.2.csv")
    
    if os.name == 'nt': # Windows
        path_to_log = Path(f"C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\TranslucentActivityRelationships-main\\new_eval\\translucent_datasets\\{LOG_NAME}\\{LOG_NAME}_0.2.csv")
    
    # Import the log
    base_log = pd.read_csv(path_to_log)
    
    func = partial(evaluate_noise_type, base_log=base_log)

    #evaluate_noise_type(False, base_log)
    
    
    # 3. Use the pool
    num_processes = min(mp.cpu_count(), len(noise_types))
    
    with mp.Pool(processes=num_processes) as pool:
        # pool.map automatically passes each item in noise_types as the 
        # first argument to 'func', while base_log remains fixed.
        pool.map(func, noise_types)
    
    print( LOG_NAME + ": Petrify evaluation completed for all noise types.")
    
    


