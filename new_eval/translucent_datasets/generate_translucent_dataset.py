# Script generates translucent logs with noise with different filter paramters
import pandas as pd
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

def variant_ratio(
    df: pd.DataFrame,
    case_col: str = "case:concept:name",
    activity_col: str = "concept:name",
) -> float:
    """
    Calculate the variant ratio of an event log.

    Variant ratio = number of distinct trace variants / total number of traces.
    - 0.0 → all traces are identical
    - 1.0 → every trace is unique

    Parameters
    ----------
    df :           Event log as a pandas DataFrame
    case_col :     Column identifying the case/process instance
    activity_col : Column identifying the activity name
    """
    traces = df.groupby(case_col, sort=False)[activity_col].apply(tuple)
    return traces.nunique() / len(traces)

def translucent_variant_ratio(
    df: pd.DataFrame,
    case_col: str = "case:concept:name",
    activity_col: str = "concept:name",
    enabled_col: str = "enabled_activities",
) -> float:
    """
    Calculate the variant ratio of a translucent event log.

    A translucent trace is represented as a sequence of (activity, frozenset(enabled))
    pairs. Two traces are the same variant only if both the executed activities
    AND the enabled sets match at every position.

    Variant ratio = number of distinct translucent variants / total number of traces.
    - 0.0 → all traces are identical (same activities + same enabled sets)
    - 1.0 → every trace is unique

    Parameters
    ----------
    df :           Translucent event log as a pandas DataFrame
    case_col :     Column identifying the case/process instance
    activity_col : Column identifying the executed activity
    enabled_col :  Column containing the enabled activities (iterable per event)
    """
    def to_translucent_trace(group):
        return tuple(
            (act, frozenset(enabled))
            for act, enabled in zip(group[activity_col], group[enabled_col])
        )

    traces = df.groupby(case_col, sort=False).apply(to_translucent_trace)
    return traces.nunique() / len(traces)

if __name__ == "__main__":

    dataset_path="/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main/new_eval/original_datasets/Sepsis Cases - Event Log.xes.gz"
    if os.name == 'nt': # Windows
        dataset_path = r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\original_datasets\Hospital Billing - Event Log.xes.gz"
    dataset_name='hospital_billing'
    input_log_path = "/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main/new_eval/translucent_datasets/road_traffic_fine/road_traffic_fine_0.2.csv"
    model_noise_threshold_list=[0.2]
    noise_types = ["add_enabled", "remove_enabled", "change_events"]


    for model_noise_threshold in model_noise_threshold_list:
        if not input_log_path:
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
            
            
        # Write the log without noise
            output_log_path = f"/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main/new_eval/translucent_datasets/{dataset_name}/{dataset_name}_{model_noise_threshold}.csv"
            if os.name == 'nt': # Windows
                output_log_path = f"C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\TranslucentActivityRelationships-main\\new_eval\\translucent_datasets\\{dataset_name}\\{dataset_name}_{model_noise_threshold}.csv"
            df_log = pm4py.convert_to_dataframe(annotated_log)
            df_log.to_csv(output_log_path, index=False)
            print(f'Generated translucent log with noise threshold {model_noise_threshold} at {output_log_path}')
        
        else:
            df_log = pd.read_csv(input_log_path)
        
        # Write the noisy logs
        for noise_type in noise_types:
            noisy_log = get_noisy_log(copy.deepcopy(df_log), noise_type, selection_percentage=0.1)
            output_log_path = f"/home/eliasmullers/Desktop/thesis/TranslucentActivityRelationships-main/new_eval/translucent_datasets/{dataset_name}/{dataset_name}_{model_noise_threshold}_{noise_type}.csv"
            if os.name == 'nt': # Windows
                output_log_path = f"C:\\Users\\elias\\Masterarbeit_code\\Spielplatz\\Code_Harry\\TranslucentActivityRelationships-main\\new_eval\\translucent_datasets\\{dataset_name}\\{dataset_name}_{model_noise_threshold}_{noise_type}.csv"
            noisy_log.to_csv(output_log_path, index=False)
            print(f'Generated translucent log with noise type {noise_type} and noise threshold {model_noise_threshold} at {output_log_path}')
            print(f'Variant ratio of the noisy log: {variant_ratio(noisy_log)}')
            print(f'Translucent variant ratio of the noisy log: {translucent_variant_ratio(noisy_log)}')
            print()


    

