from translucent_fitness.evaluate.quantitative.adding.add_enabled import manipulate_log_optimized as add_enabled
from translucent_fitness.evaluate.quantitative.adding.add_events import manipulate_log_optimized as add_events
from translucent_fitness.evaluate.quantitative.removing.create_logs import manipulate_log_optimized as remove_enabled
from translucent_fitness.evaluate.quantitative.changing.create_logs import manipulate_log_optimized as change_events

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
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")