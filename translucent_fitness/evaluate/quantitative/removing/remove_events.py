import pandas as pd
import numpy as np

# Removes a percentage of events from a given log
def manipulate_log_optimized(df, selection_percentage):
    # Set random seeds for reproducibility
    np.random.seed(42)

    # Calculate number of rows to drop based on selection_percentage
    num_rows_to_select = int(len(df) * selection_percentage)

    # Randomly select rows WITHOUT replacement
    selected_indices = np.random.choice(df.index, size=num_rows_to_select, replace=False)

    # Drop all selected indices at once in a vectorized operation
    df_cleaned = df.drop(index=selected_indices)

    return df_cleaned