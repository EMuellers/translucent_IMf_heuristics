# Script generates translucent logs with noise with different filter paramters

import os
import time
import sys
sys.path.append(r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main")


import pm4py

from new_eval.generate_log import generate_log_without_noise

start = time.time()

dataset_path=r"C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\TranslucentActivityRelationships-main\\evaluation\\sepsis\\Sepsis Cases - Event Log.xes.gz"
dataset_name='Sepsis'
model_noise_threshold_list=[0.2]

# Create output directory if it doesn't exist
output_dir = os.path.join('', dataset_name)
os.makedirs(output_dir, exist_ok=True)

for model_noise_threshold in model_noise_threshold_list:
    annotated_log = generate_log_without_noise(dataset_path,
                                            model_noise_threshold,
                                            {},
                                            enabled_activities_name="enabled_activities")
    #output_log_path = os.path.join(output_dir, f'{dataset_name}_{model_noise_threshold}.xes')
    #pm4py.write_xes(annotated_log, output_log_path)
    output_log_path = os.path.join(output_dir, f'{dataset_name}_{model_noise_threshold}.csv')
    df_log = pm4py.convert_to_dataframe(annotated_log)
    df_log.to_csv(output_log_path, index=False)
    print(f'Generated translucent log with noise threshold {model_noise_threshold} at {output_log_path}')
    print(f'Time taken so far: {time.time() - start} seconds')
