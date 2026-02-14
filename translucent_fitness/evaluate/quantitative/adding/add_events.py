import pandas as pd
import numpy as np


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
                'enabled_activities': ','.join(enabled_activities_sample)  # Convert list back to comma-separated string
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
