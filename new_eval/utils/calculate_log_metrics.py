import pandas as pd
import pm4py
from translucent_discovery.translucent_inductive_miner.translucent_datatype import translucent_log_to_tcl

def calculate_pm4py_metrics(file_path, case_id_col='case:concept:name', activity_col='concept:name', timestamp_col='time:timestamp'):
    """
    Reads a CSV and calculates metrics using pm4py.
    """
    try:
        # 1. Load the CSV into a pandas DataFrame first
        df = pd.read_csv(file_path)
        
        # 2. Format the DataFrame so pm4py understands it.
        # This renames your custom columns to standard XES format: 
        # (case:concept:name, concept:name, time:timestamp)
        df = pm4py.format_dataframe(df, case_id=case_id_col, activity_key=activity_col, timestamp_key=timestamp_col)
        
        # 3. Convert the DataFrame into a formal pm4py Event Log object
        # This groups the events into proper Trace objects based on the timestamp order
        log = pm4py.convert_to_event_log(df)
        
        # --- Calculate Metrics ---
        
        tcl = translucent_log_to_tcl(log)
        
        total_translucent_variants = len(tcl.keys())
        
        # Total Traces (cases) is just the length of the log
        total_traces = len(log)
        
        # Total Events can be summed up by counting events in each trace
        total_events = sum(len(trace) for trace in log)
        
        # PM4Py has a built-in function to extract variants
        variants = pm4py.get_variants(log)
        total_variants = len(variants)
        
        # PM4Py has a built-in function to get unique event/activity types
        # We look up the standard 'concept:name' attribute which pm4py uses for activities
        event_types = pm4py.get_event_attribute_values(log, "concept:name")
        total_event_types = len(event_types)
        
        # Trace Lengths: Extract lengths and find min/max/avg
        trace_lengths = [len(trace) for trace in log]
        min_trace_length = min(trace_lengths) if trace_lengths else 0
        max_trace_length = max(trace_lengths) if trace_lengths else 0
        avg_trace_length = sum(trace_lengths) / total_traces if total_traces > 0 else 0
        
        # Output the results
        print("--- Event Log Metrics (PM4Py) ---")
        print(f"{total_events} Events")
        print(f"{total_traces} Traces")
        print(f"{total_variants} Variants")
        print(f"{total_translucent_variants} Translucent Variants")
        print(f"{total_event_types} Event Types (activity types)")
        print(f"min trace length: {min_trace_length}")
        print(f"max trace length: {max_trace_length}")
        print(f"avg. trace length: {avg_trace_length:.2f}")

    except FileNotFoundError:
        print(f"Error: Could not find the file at '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # Ensure you have pm4py installed: pip install pm4py
    # Replace these with your actual file path and column headers
    csv_path = r"C:\Users\elias\Masterarbeit_code\Spielplatz\Code_Harry\TranslucentActivityRelationships-main\new_eval\translucent_datasets\hospital_billing\hospital_billing_remove_enabled.csv" 
    
    calculate_pm4py_metrics(csv_path)