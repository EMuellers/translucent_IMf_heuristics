from translucent_fitness.evaluate.quantitative.adding.add_enabled import manipulate_log_optimized as add_enabled
from translucent_fitness.evaluate.quantitative.adding.add_events import manipulate_log_optimized as add_events
from translucent_fitness.evaluate.quantitative.removing.create_logs import manipulate_log_optimized as remove_enabled
from translucent_fitness.evaluate.quantitative.changing.create_logs import manipulate_log_optimized as change_events
from translucent_fitness.evaluate.quantitative.adding.add_events_no_translucent import manipulate_log_optimized as add_events_no_translucent
from translucent_fitness.evaluate.quantitative.removing.remove_events import manipulate_log_optimized as remove_events

def get_noisy_log(log, noise_type, selection_percentage=0.2):
    """
    log: pandas DataFrame event log
    selection_percentage: percentage of traces to which noise should be added
    """
    if noise_type == "add_enabled":
        return add_enabled(log, selection_percentage)
    elif noise_type == "add_events":
        return add_events(log, selection_percentage)
    elif noise_type == "remove_enabled":
        return remove_enabled(log, selection_percentage)
    elif noise_type == "change_events":
        return change_events(log, selection_percentage)
    elif noise_type == "add_events_no_trans":
        return add_events_no_translucent(log, selection_percentage)
    elif noise_type == "remove_events":
        return remove_events(log, selection_percentage)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import copy
    noise_types = ["remove_events"]
    log_names = ["hospital_billing", "Sepsis", "road_traffic_fine"]
    for log_name in log_names:
        log = pd.read_csv(rf"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\translucent_datasets\{log_name}\{log_name}_base.csv")
        for noise_type in noise_types:
            noisy_log = get_noisy_log(copy.deepcopy(log), noise_type=noise_type, selection_percentage=0.1)
            noisy_log.to_csv(rf"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\translucent_datasets\{log_name}\{log_name}_{noise_type}.csv", index=False)
