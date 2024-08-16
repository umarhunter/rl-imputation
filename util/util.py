import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from stable_baselines3.common.evaluation import evaluate_policy


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


def result_handler(model, env):
    # Assuming env is a DummyVecEnv
    env = env.envs[0]  # Unwrap the original environment

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    if env.counter > 0:
        print("Counter: ", env.counter)

    # Save the trained model
    model.save('results/dqn_imputation_model')

    # Assuming env is your environment and you've already run the agent
    imputed_data = env.incomplete_data  # Data after imputation
    complete_data = env.complete_data  # The original complete data

    # Ensure the indices of missing values are the same as those used during training
    missing_indices = env.missing_indices

    # Initialize lists to store the actual and imputed values
    actual_values = []
    imputed_values = []

    # Tolerance level for considering values as matches
    tolerance = 0.05

    # Extract the actual and imputed values at the missing indices
    for row, col in missing_indices:
        actual_values.append(complete_data.iloc[row, col])
        imputed_values.append(imputed_data.iloc[row, col])

    # Convert lists to numpy arrays for comparison
    actual_values = np.array(actual_values)
    imputed_values = np.array(imputed_values)

    # Calculate the Mean Absolute Error (MAE) or Mean Squared Error (MSE)
    mae = mean_absolute_error(actual_values, imputed_values)
    mse = mean_squared_error(actual_values, imputed_values)

    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")

    # Calculate the percentage of values that match within the tolerance
    tolerance_match_rate = np.mean(np.abs(actual_values - imputed_values) <= tolerance) * 100
    print(f"Tolerance-Based Match Rate (±{tolerance}): {tolerance_match_rate:.2f}%")
