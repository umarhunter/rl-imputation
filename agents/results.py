import pandas as pd
import numpy as np

def calculate_mae(df1, df2):
    """Calculate Mean Absolute Error (MAE) between two DataFrames."""
    if df1.shape != df2.shape:
        raise ValueError("DataFrames do not have the same shape")
    return np.mean(np.abs(df1 - df2))

def calculate_mse(df1, df2):
    """Calculate Mean Squared Error (MSE) between two DataFrames."""
    if df1.shape != df2.shape:
        raise ValueError("DataFrames do not have the same shape")
    return np.mean((df1 - df2) ** 2)

def calculate_rmse(df1, df2):
    """Calculate Root Mean Squared Error (RMSE) between two DataFrames."""
    if df1.shape != df2.shape:
        raise ValueError("DataFrames do not have the same shape")
    return np.sqrt(calculate_mse(df1, df2))

def compare_dataframes(df1, df2):
    """Compare two DataFrames and return the indices and values where they differ."""
    if df1.shape != df2.shape:
        raise ValueError("DataFrames do not have the same shape")

    # Find the differences
    diff = df1 != df2

    # Get the indices where the differences occur
    diff_indices = np.where(diff)

    # Create a DataFrame to store the differences
    differences = pd.DataFrame({
        'Row': diff_indices[0],
        'Column': df1.columns[diff_indices[1]],
        'Value_df1': df1.values[diff_indices],
        'Value_df2': df2.values[diff_indices]
    })

    return differences

original_df = pd.read_csv("breast_cancer_wisconsin.csv")
target_column = ['ID', 'Diagnosis']
original_df = original_df.drop(columns=target_column)
imputed_df = pd.read_csv("checkpoints/checkpoint_250.csv")

differences = compare_dataframes(original_df, imputed_df)
print(differences)

mae = calculate_mae(original_df, imputed_df)
mse = calculate_mse(original_df, imputed_df)
rmse = calculate_rmse(original_df, imputed_df)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")  