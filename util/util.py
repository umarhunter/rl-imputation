import logging
import numpy as np
import pandas as pd
import json
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error
from stable_baselines3.common.evaluation import evaluate_policy
from numpy import float32


def get_dataset_name(dataset_id):
    dataset_names = {
        94: "spambase",
        59: "letter_recognition",
        17: "breast_cancer_wisconsin",
        332: "online_news_popularity",
        350: "default_credit_card_clients",
        189: "parkinsons_telemonitoring",
        484: "travel_reviews",
        149: "statlog_vehicle_silhouettes"
    }
    return dataset_names.get(dataset_id, "unknown_dataset")


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
            df_with_missing.iloc[:, col_idx] = df_with_missing.iloc[:, col_idx].astype(float32)

        # Set NaN for the chosen index
        df_with_missing.iat[row_idx, col_idx] = np.nan

    return df_with_missing


def calculate_errors(imputed_data, actual_data):
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(imputed_data.values - actual_data.values))

    # Calculate Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((imputed_data.values - actual_data.values) ** 2))

    return mae, rmse


def get_decimal_places(value):
    """
    Returns the number of decimal places in a value.
    """
    if pd.isna(value):  # Handle NaN values
        return 0
    if isinstance(value, float):
        decimals = abs(np.floor(np.log10(np.abs(value)))) if value != 0 else 0
        return int(decimals)
    return 0


def truncate_to_match_original(imputed_data, complete_data_original):
    """
    Truncates the imputed values to match the precision of the original values in the complete_data.
    Both dataframes should have the same shape and column names.
    """
    for row_idx in range(imputed_data.shape[0]):
        for col_idx, col_name in enumerate(imputed_data.columns):
            # Get the original value from the complete data
            original_value = complete_data_original.iloc[row_idx, col_idx]

            # Determine the precision of the original value
            precision = get_decimal_places(original_value) + 1

            # Truncate the imputed value to match the original value's precision
            imputed_value = imputed_data.iloc[row_idx, col_idx]
            truncated_value = np.floor(imputed_value * 10 ** precision) / 10 ** precision
            imputed_data.iloc[row_idx, col_idx] = truncated_value

    return imputed_data


def result_handler(model, env, dataset_id, episodes):
    """
    Handles the result after training the model by evaluating performance,
    saving the imputed data in its original scale, and saving metrics in a JSON file.
    """
    # Assuming env is a DummyVecEnv
    env = env.envs[0]  # Unwrap the original environment

    # Evaluate the model after training
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    counter = env.unwrapped.counter if hasattr(env.unwrapped, 'counter') else None
    if counter and counter > 0:
        print(f"Counter: {counter}")

    # Once training is done, inverse-transform the imputed data back to the original scale
    imputed_data_original_scale = env.unwrapped.finalize_imputation()

    imputed_data_original_scale = truncate_to_match_original(imputed_data_original_scale,
                                                             env.unwrapped.complete_data_original)

    # Debug: Check if rounding is happening
    print("First 5 rows of imputed data after rounding:")
    print(imputed_data_original_scale.head())

    print("First 5 rows of complete data:")
    print(env.unwrapped.complete_data_original.head())

    env.unwrapped.complete_data_original.to_csv('complete_data_original.csv', index=False)
    # Compare with the original complete data
    comparison = imputed_data_original_scale.compare(env.unwrapped.complete_data_original)

    # Compute metrics
    mae = mean_absolute_error(env.unwrapped.complete_data_original.values, imputed_data_original_scale.values)
    mse = mean_squared_error(env.unwrapped.complete_data_original.values, imputed_data_original_scale.values)

    # adjsting tolerance to 0.001
    tolerance = 0.001  # 0.1% tolerance
    tolerance_match_rate = np.mean(
        np.abs(env.unwrapped.complete_data_original.values - imputed_data_original_scale.values) <= tolerance
    ) * 100

    # Print and log the evaluation metrics
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"Tolerance-Based Match Rate (±{tolerance}): {tolerance_match_rate:.2f}%")

    # Calculate similarity percentage
    similarity_percentage = 100 * (1 - comparison.shape[0] / imputed_data_original_scale.size)
    print(f"Similarity Percentage: {similarity_percentage:.2f}%")

    # Create result data for JSON output
    result_data = {
        'dataset_name': get_dataset_name(dataset_id),
        'MAE': mae,
        'MSE': mse,
        'mean_reward': mean_reward,
        'counter': counter,
        'tolerance_match_rate': tolerance_match_rate,
        'total_timesteps': episodes,
        'num_actions': env.unwrapped.num_actions,
        'similarity_percentage': similarity_percentage
    }

    # Ensure unique file names by appending a numerical suffix
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    base_file_name = f"{result_data['dataset_name']}"
    file_index = 1

    while True:
        file_name = f"{base_file_name}_{file_index}.json"
        file_name_csv = f"{base_file_name}_{file_index}.csv"
        file_path = os.path.join(output_dir, file_name)
        file_name_csv = os.path.join(output_dir, file_name_csv)
        if not os.path.exists(file_path):
            break
        file_index += 1

    # Save the results as a JSON file with a unique name
    with open(file_path, 'w') as f:
        json.dump(result_data, f, indent=4)

    imputed_data_original_scale.to_csv(file_name_csv, index=False)
    logging.info(f"JSON results saved to {file_path}")
    logging.info(f"Imputed CSV data saved to {file_name_csv}")
