# Script generates translucent logs with noise with different filter paramters

import os
import copy
import sys
if os.name == 'nt': # Windows
    sys.path.append(r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main")
else: # Linux
    sys.path.append("/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main")

import pm4py

from new_eval.generate_log import generate_log_without_noise
from new_eval.translucent_datasets.generate_noisy_datasets import get_noisy_log


dataset_path="/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main/new_eval/original_datasets/Sepsis Cases - Event Log.xes.gz"
dataset_name='Sepsis'
model_noise_threshold_list=[0.2]
noise_types = ["add_enabled", "add_events", "remove_enabled", "change_events"]


for model_noise_threshold in model_noise_threshold_list:
    annotated_log = generate_log_without_noise(dataset_path,
                                            model_noise_threshold,
                                            {},
                                            enabled_activities_name="enabled_activities")
    
    # Check correctness: Is the excecuted activity always included in the enabled activities?
    for trace in annotated_log:
        for i in range(len(trace)):
            event = trace[i]
            executed_activity = event["concept:name"]
            enabled_activities = set(event["enabled_activities"].split(', '))
            if executed_activity not in enabled_activities:
                print(f"Error: Executed activity '{executed_activity}' is not in the set of enabled activities {enabled_activities} for event {i} in trace {trace.attributes['concept:name']}")
    
    # Write the noisy logs
    for noise_type in noise_types:
        noisy_log = get_noisy_log(copy.deepcopy(annotated_log), noise_type)
        output_log_path = f"/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main/new_eval/translucent_datasets/{dataset_name}/{dataset_name}_{model_noise_threshold}_{noise_type}.csv"
        df_log = pm4py.convert_to_dataframe(noisy_log)
        df_log.to_csv(output_log_path, index=False)
        print(f'Generated translucent log with noise type {noise_type} and noise threshold {model_noise_threshold} at {output_log_path}')

    
    # Write the log without noise
    output_log_path = f"/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main/new_eval/translucent_datasets/{dataset_name}/{dataset_name}_{model_noise_threshold}.csv"
    df_log = pm4py.convert_to_dataframe(annotated_log)
    df_log.to_csv(output_log_path, index=False)
    print(f'Generated translucent log with noise threshold {model_noise_threshold} at {output_log_path}')
