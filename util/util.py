import numpy as np
import pandas as pd


def generate_missing_df(df, missing_rate):
    """Introduce missing values randomly into the dataframe at the specified rate."""
    df_with_missing = df.copy()

    # Total number of elements in the dataframe
    total_elements = df_with_missing.size

    # Number of elements to be set as NaN
    num_missing = int(missing_rate * total_elements)

    # Flatten the DataFrame to get flat indices
    flat_indices = np.arange(total_elements)

    # Get random indices
    missing_indices = np.random.choice(flat_indices, num_missing, replace=False)

    # Convert the flat indices to multi-dimensional indices
    multi_dim_indices = np.unravel_index(missing_indices, df_with_missing.shape)

    # Assign NaN to the chosen indices
    for row_idx, col_idx in zip(*multi_dim_indices):
        if pd.api.types.is_integer_dtype(df_with_missing.iloc[:, col_idx]):
            # Convert integer column to float first if necessary
            df_with_missing.iloc[:, col_idx] = df_with_missing.iloc[:, col_idx].astype(float)

        # Set NaN for the chosen index
        df_with_missing.iat[row_idx, col_idx] = np.nan

    return df_with_missing



def calculate_errors(imputed_data, actual_data):
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(imputed_data.values - actual_data.values))

    # Calculate Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((imputed_data.values - actual_data.values) ** 2))

    return mae, rmse
