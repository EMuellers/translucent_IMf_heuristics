import pandas as pd
import numpy as np
import random


def manipulate_log(df, addition_percentage):
    # Set a random seed for reproducibility
    np.random.seed(42)


    all_activities = set(df['concept:name'].unique())

    def add_new_rows(existing_df, num_new_rows):
        # Prepare a list to hold new rows
        new_rows = []

        # Define probabilities for number of enabled activities (1 through len(all_activities))
        max_enabled_activities = len(list(all_activities))
        choices = np.arange(1, max_enabled_activities + 1)

        # Assign weights: higher weight for lower numbers
        weights = np.linspace(2, 1, max_enabled_activities)

        temp = weights / weights.sum()
        print(temp)

        for _ in range(num_new_rows):
            # Randomly select an index from existing DataFrame
            random_index = np.random.choice(existing_df.index)

            # Get caseid and timestamp from the randomly selected row
            selected_caseid = existing_df['case:concept:name'].iloc[random_index]
            selected_timestamp = existing_df['time:timestamp'].iloc[random_index]

            # Select number of enabled activities based on weighted choices
            num_enabled_activities = np.random.choice(choices, p=weights / weights.sum())

            enabled_activities_sample = np.random.choice(list(all_activities), size=num_enabled_activities,
                                                         replace=False)

            # Select one executed activity randomly from enabled activities
            executed_activity = np.random.choice(enabled_activities_sample)

            # Create a dictionary representing the new row/event
            new_row = {
                'case:concept:name': selected_caseid,
                'concept:name': executed_activity,
                'time:timestamp': selected_timestamp,
                'enabled_activities': ', '.join(enabled_activities_sample)  # Convert list back to comma-separated string
            }

            # Append this dictionary to our list of new rows
            new_rows.append(new_row)

        # Convert list of dictionaries into a DataFrame and append it to existing DataFrame
        new_rows_df = pd.DataFrame(new_rows)
        updated_df = pd.concat([existing_df, new_rows_df], ignore_index=True)

        return updated_df

    # Calculate number of new rows to add based on addition_percentage
    num_new_rows_to_add = int(len(df) * addition_percentage)

    df_updated = add_new_rows(df, num_new_rows_to_add)

    # Save modified DataFrame back to a new CSV file (optional)
    #df_updated.to_csv('log_after_adding_enabled.csv', index=False)

    return df_updated

def manipulate_log_optimized(df, addition_percentage):
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Calculate number of new rows to add
    num_new_rows = int(len(df) * addition_percentage)
    if num_new_rows == 0:
        return df

    # OPTIMIZATION 1: Pre-calculate lists and probabilities ONCE
    all_activities_list = list(df['concept:name'].unique())
    max_enabled_activities = 1

    choices = np.arange(1, max_enabled_activities + 1)
    weights = np.linspace(2, 1, max_enabled_activities)
    probabilities = weights / weights.sum()

    # OPTIMIZATION 2: Extract columns to NumPy arrays for instant lookup
    # This completely eliminates the slow .iloc[] pandas overhead
    case_ids = df['case:concept:name'].values
    timestamps = df['time:timestamp'].values

    # Bulk generate ALL random indices and activity counts in one go
    random_indices = np.random.choice(len(df), size=num_new_rows, replace=False)
    num_enabled_arr = np.random.choice(choices, size=num_new_rows, p=probabilities)

    # Fast NumPy array indexing to get all selected caseids and timestamps instantly
    selected_caseids = case_ids[random_indices]
    selected_timestamps = timestamps[random_indices]

    # Prepare lists for the string generation
    enabled_activities_list = []
    executed_activities_list = []

    # OPTIMIZATION 3: Fast pure-Python loop for the random selections
    for num_enabled in num_enabled_arr:
        # random.sample is much faster than np.random.choice for sampling without replacement
        sampled_acts = random.sample(all_activities_list, num_enabled)
        executed_act = random.choice(sampled_acts)

        enabled_activities_list.append(executed_act) # Change: Instead of adding all enabled activities, we only add the executed activity as enabled, to cause more trouble for DFG cut detection vs tdfg
        executed_activities_list.append(executed_act)

    # OPTIMIZATION 4: Construct the new DataFrame directly from lists/arrays
    new_rows_df = pd.DataFrame({
        'case:concept:name': selected_caseids,
        'concept:name': executed_activities_list,
        'time:timestamp': selected_timestamps,
        'enabled_activities': enabled_activities_list
    })

    # Concatenate and return
    updated_df = pd.concat([df, new_rows_df], ignore_index=True)
    
    updated_df['time:timestamp'] = pd.to_datetime(updated_df['time:timestamp'], utc=True)
    
    return updated_df

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\elias\Desktop\Results_22.02\TranslucentActivityRelationships-main\new_eval\translucent_datasets\hospital_billing\hospital_billing_0.2.csv")
    df_updated = manipulate_log_optimized(df, addition_percentage=0.2)
    df_updated.to_csv(r"C:\Users\elias\Desktop\Results_22.02\TranslucentActivityRelationships-main\new_eval\translucent_datasets\hospital_billing\hospital_billing_0.2_add_events.csv", index=False)