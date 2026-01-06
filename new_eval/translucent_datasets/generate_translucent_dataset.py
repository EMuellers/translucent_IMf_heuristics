# Script generates translucent logs with noise with different filter paramters

import os
import time
import sys
sys.path.append(r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main")


import pm4py

from new_eval.generate_log import generate_log_with_noise

start = time.time()

dataset_path=r"C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\TranslucentActivityRelationships-main\\evaluation\\sepsis\\Sepsis Cases - Event Log.xes.gz"
dataset_name='Sepsis'
model_noise_threshold_list=[0.2, 0.4, 0.6, 0.8, 1.0] #TODO: Für n verschiedene datasets in generate_log n random samples per trace durchführen? Dann muss nicht immer wieder der DFA generiert werden.

# Create output directory if it doesn't exist
output_dir = os.path.join('new_eval', 'translucent_datasets', dataset_name)
os.makedirs(output_dir, exist_ok=True)

for model_noise_threshold in model_noise_threshold_list:
    annotated_log = generate_log_with_noise(dataset_path,
                                            model_noise_threshold,
                                            {},
                                            enabled_activities_name="enabled_activities")
    output_log_path = os.path.join(output_dir, f'{dataset_name}_{model_noise_threshold}.xes')
    pm4py.write_xes(annotated_log, output_log_path)
    print(f'Generated translucent log with noise threshold {model_noise_threshold} at {output_log_path}')
    print(f'Time taken so far: {time.time() - start} seconds')
