# Script generates translucent logs with noise with different filter paramters

import os
import time
import sys
#sys.path.append(r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main")
sys.path.append("/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main")

import pm4py

from new_eval.generate_log import generate_log_without_noise

start = time.time()

#dataset_path=r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\original_datasets\Hospital Billing - Event Log.xes.gz"
#dataset_path="/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main/new_eval/original_datasets/Hospital Billing - Event Log.xes.gz"
#dataset_name='hospital_billing'
#dataset_path=r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\original_datasets\Hospital Billing - Event Log.xes.gz"
dataset_path=r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\original_datasets\Road_Traffic_Fine_Management_Process.xes.gz"
dataset_name='Road_Traffic_Fine_Management_Process'
model_noise_threshold_list=[0.2]

# Create output directory if it doesn't exist
#output_dir = os.path.join('', dataset_name)
#os.makedirs(output_dir, exist_ok=True)

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
    
    
    #output_log_path = os.path.join(output_dir, f'{dataset_name}_{model_noise_threshold}.xes')
    #pm4py.write_xes(annotated_log, output_log_path)
    #output_log_path = os.path.join(output_dir, f'{dataset_name}_{model_noise_threshold}.csv')
    output_log_path = "/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main/new_eval/translucent_datasets/hospital_billing/hospital_billing_0.2.csv"
    output_log_path = r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\translucent_datasets\road_traffic_fine\road_traffic_fine_0.2.csv"
    df_log = pm4py.convert_to_dataframe(annotated_log)
    df_log.to_csv(output_log_path, index=False)
    print(f'Generated translucent log with noise threshold {model_noise_threshold} at {output_log_path}')
    print(f'Time taken so far: {time.time() - start} seconds')
