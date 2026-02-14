import pandas as pd
import numpy as np


def manipulate_log(df, selection_percentage):
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Calculate number of rows to select based on selection_percentage
    num_rows_to_select = int(len(df) * selection_percentage)

    # Randomly select rows with replacement
    selected_indices = np.random.choice(df.index, size=num_rows_to_select, replace=True)

    all_activities = set(df['concept:name'].unique())

    def modify_selected_row(row):
        enabled_activities = [el.strip() for el in row['enabled_activities'].split(',')]

        # Convert to set to remove duplicates and then back to list for processing
        unique_enabled_activities = set(enabled_activities)

        if len(enabled_activities) > 0:
            new_activity = str(np.random.choice(list(all_activities)))
            unique_enabled_activities.add(new_activity)
            row['enabled_activities'] = ','.join(unique_enabled_activities)

        return row

    # Apply modifications only to selected rows
    for index in selected_indices:
        df.loc[index] = modify_selected_row(df.loc[index])


    # Save modified DataFrame back to a new CSV file (optional)
    #df.to_csv('log_after_adding_enabled.csv', index=False)
    return df

