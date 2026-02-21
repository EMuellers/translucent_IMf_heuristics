import pandas as pd
import numpy as np
import random

def manipulate_log(df, selection_percentage):
    # Set a random seed for reproducibility
    np.random.seed(42)

    # Calculate number of rows to select based on selection_percentage
    num_rows_to_select = int(len(df) * selection_percentage)

    # Randomly select rows with replacement
    selected_indices = np.random.choice(df.index, size=num_rows_to_select, replace=True)

    # Create a set to keep track of indices to drop
    indices_to_drop = set()

    def modify_selected_row(row):
        enabled_activities = [el.strip() for el in row['enabled_activities'].split(',')]

        if len(enabled_activities) > 0:
            # Randomly choose an activity to remove from enabled activities
            activity_to_remove = np.random.choice(enabled_activities)

            # Check if the removed activity is the same as the one in 'activity' column
            if str(activity_to_remove) == str(row['concept:name']):
                return True  # Mark for deletion

            # Remove selected activity from enabled_activities list
            remaining_activities = [act for act in enabled_activities if act != activity_to_remove]

            # Update remaining enabled activities back to string format
            row['enabled_activities'] = ', '.join(remaining_activities)

        return False  # Row remains valid

    # Apply modifications only to selected rows and collect indices for removal
    for index in selected_indices:
        if modify_selected_row(df.loc[index]):
            indices_to_drop.add(index)

    # Drop marked indices from DataFrame
    df_cleaned = df.drop(index=indices_to_drop)

    # Save cleaned DataFrame back to a new CSV file (optional)
    return df_cleaned

def manipulate_log_optimized(df, selection_percentage):
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Calculate number of rows to select based on selection_percentage
    num_rows_to_select = int(len(df) * selection_percentage)

    # Randomly select rows WITHOUT replacement (prevents duplicate operations on same row)
    selected_indices = np.random.choice(df.index, size=num_rows_to_select, replace=False)

    indices_to_drop = set()

    # OPTIMIZATION 1: Extract columns to fast pure Python dictionaries
    enabled_activities_dict = df['enabled_activities'].to_dict()
    concept_name_dict = df['concept:name'].to_dict()

    # OPTIMIZATION 2: Process modifications and drop-checks in pure Python
    for index in selected_indices:
        current_activities_str = enabled_activities_dict[index]
        
        # Guard against NaN/float values
        if not isinstance(current_activities_str, str):
            continue
            
        enabled_activities = [el.strip() for el in current_activities_str.split(',')]

        if len(enabled_activities) > 0:
            # OPTIMIZATION 3: random.choice is much faster for standard Python lists
            activity_to_remove = str(random.choice(enabled_activities))

            # Check if the removed activity is the same as the one in 'concept:name'
            if activity_to_remove == str(concept_name_dict[index]):
                indices_to_drop.add(index)  # Mark for deletion
            else:
                # Remove selected activity from the list
                remaining_activities = [act for act in enabled_activities if act != activity_to_remove]
                
                # Update the dictionary directly
                enabled_activities_dict[index] = ', '.join(remaining_activities)

    # OPTIMIZATION 4: Bulk update the DataFrame and drop rows in vectorized operations
    # Write the modified dictionary back to the DataFrame
    df['enabled_activities'] = df.index.map(enabled_activities_dict)

    # Drop all marked indices at once
    df_cleaned = df.drop(index=list(indices_to_drop))

    return df_cleaned