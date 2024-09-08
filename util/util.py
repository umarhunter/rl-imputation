import logging
import numpy as np
import pandas as pd
import json
import os

from sklearn.metrics import mean_absolute_error, mean_squared_error
from stable_baselines3.common.evaluation import evaluate_policy
from numpy import float64

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
            df_with_missing.iloc[:, col_idx] = df_with_missing.iloc[:, col_idx].astype(float64)

        # Set NaN for the chosen index
        df_with_missing.iat[row_idx, col_idx] = np.nan

    return df_with_missing


def calculate_errors(imputed_data, actual_data):
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(imputed_data.values - actual_data.values))

    # Calculate Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((imputed_data.values - actual_data.values) ** 2))

    return mae, rmse


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

    # Convert the final imputed data back to a DataFrame (if necessary)
    imputed_data_original_scale = pd.DataFrame(imputed_data_original_scale, columns=env.unwrapped.incomplete_data.columns)

    # Save the final imputed dataset in the original scale
    imputed_data_original_scale.to_csv(f"results/imputed_data_original_{dataset_id}.csv", index=False)

    # Compare with the original complete data
    comparison = imputed_data_original_scale.compare(env.unwrapped.complete_data)

    # Calculate Mean Absolute Error (MAE), Mean Squared Error (MSE), and Tolerance Match Rate
    mae = mean_absolute_error(env.unwrapped.complete_data.values, imputed_data_original_scale.values)
    mse = mean_squared_error(env.unwrapped.complete_data.values, imputed_data_original_scale.values)
    tolerance = 0.05
    tolerance_match_rate = np.mean(np.abs(env.unwrapped.complete_data.values - imputed_data_original_scale.values) <= tolerance) * 100

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


