import logging
import random
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
import csv

from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

# Configure logging to write to both console and file
log_filename = "training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(log_filename, mode='w')  # Output to file
    ]
)

def calculate_metrics(env, agent):
    # Get the imputed data (after training)
    imputed_data = env.state

    # Check for NaN values in complete_data and imputed_data
    if env.complete_data.isna().sum().sum() > 0:
        nan_counts = env.complete_data.isna().sum()  # Series with NaN counts per column
        nan_columns = nan_counts[nan_counts > 0]  # Filter columns with NaNs
        logging.error("NaN values found in complete_data:")
        for col, count in nan_columns.items():
            logging.error(f"Complete Data: Column '{col}': {count} NaN values")
        return None, None  # Early exit if complete_data has NaNs

    if imputed_data.isna().sum().sum() > 0:
        nan_counts = env.complete_data.isna().sum()  # Series with NaN counts per column
        nan_columns = nan_counts[nan_counts > 0]  # Filter columns with NaNs
        logging.error("NaN values found in complete_data:")
        for col, count in nan_columns.items():
            logging.error(f"Imputed Data: Column '{col}': {count} NaN values")
        return None, None  # Early exit if complete_data has NaNs

    # Calculate MAE and RMSE between imputed data and complete data
    mae = mean_absolute_error(env.complete_data.values.flatten(), imputed_data.values.flatten())
    rmse = root_mean_squared_error(env.complete_data.values.flatten(), imputed_data.values.flatten())

    return mae, rmse

def preprocess_columns_for_missing(df):
    """Preprocess columns by casting int64 columns to float64 to handle NaN values."""
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype(np.float64)  # Convert int64 to float64
    return df

def generate_missing_df(df, missing_rate):
    """Introduce missing values randomly into the dataframe at the specified rate."""
    df_with_missing = preprocess_columns_for_missing(df)

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

    # Assign NaN to the missing indices
    for row_idx, col_idx in zip(*multi_dim_indices):
        df_with_missing.iat[row_idx, col_idx] = np.nan

    return df_with_missing

def load_dataset(datasetid, missing_rate):
    dataset = fetch_ucirepo(id=datasetid)
    df = dataset.data.original

    # Drop the target columns before generating missing values
    target_columns = dataset.metadata.target_col
    logging.info(f"Target columns: {target_columns}")

    # Ensure target_columns is valid
    if target_columns and set(target_columns).issubset(df.columns):
        df_dropped = df.drop(columns=target_columns)
        logging.info(f"Dropped target columns: {target_columns}")
    else:
        logging.warning(f"Target columns are missing or not valid for dataset {datasetid}. Proceeding without dropping any columns.")
        df_dropped = df  # If no valid target columns, don't drop any columns

    # Drop all non-numerical columns
    df_numerical = df_dropped.select_dtypes(include=[np.number])
    logging.info(f"Retained numerical columns: {df_numerical.columns.tolist()}")

    # Use df_numerical as complete_data (without missing values)
    complete_data = df_numerical.fillna(df_numerical.mean())  # Initial fill with column means

    # Check for any columns that are still NaN and fill with 0 or another default value
    if complete_data.isna().sum().sum() > 0:
        logging.warning("Some columns in complete_data still contain NaNs after mean fill. Filling remaining NaNs with 0.")
        complete_data = complete_data.fillna(0)  # Fill any remaining NaNs with 0 as a last resort

    # Generate missing values for incomplete_data using the now fully populated complete_data
    incomplete_data = generate_missing_df(complete_data, missing_rate)  # Introduce experimental missing values

    return complete_data, incomplete_data


# Function to log results to a CSV file
def log_results(dataset_id, missing_rate, mae, rmse):
    # Ensure the results directory exists
    results_file = "./results/experiment_results.csv"
    file_exists = os.path.isfile(results_file)

    with open(results_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Dataset ID", "Missing Rate", "MAE", "RMSE"])  # Write header if file doesn't exist

        writer.writerow([dataset_id, missing_rate, mae, rmse])
    logging.info(f"Results logged for dataset {dataset_id} and missing rate {missing_rate}.")



class RLImputer:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        logging.info("Initializing Q-Learning Agent")
        self.env = env
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Simplified Q-table: store Q-values for positions and actions
        self.q_table = defaultdict(lambda: defaultdict(float))

    def choose_action(self, position):
        """Choose an action using an epsilon-greedy policy."""
        state_key = tuple(position)  # Simplify state representation

        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.env.get_possible_actions(position[1]))
            return action
        else:
            # Exploitation: choose the best known action
            actions = self.env.get_possible_actions(position[1])
            q_values = {a: self.q_table[state_key][a] for a in actions}
            best_action = max(q_values, key=q_values.get)
            return best_action

    def learn(self, position, action, reward, next_position):
        """Update the Q-table based on the action taken."""
        state_key = tuple(position)
        next_state_key = tuple(next_position)

        q_predict = self.q_table[state_key][action]
        q_target = reward + self.gamma * max(self.q_table[next_state_key].values(), default=0)
        self.q_table[state_key][action] += self.alpha * (q_target - q_predict)

    def train(self, episodes=250, log_interval=1):
        steps_per_episode = []  # Track steps for each episode
        mae_per_episode = []  # Track MAE for each episode
        rmse_per_episode = []  # Track RMSE for each episode

        mae, rmse = None, None  # Initialize in case of no episodes

        for episode in range(1, episodes + 1):
            state = self.env.reset()
            done = False
            step_count = 0  # Track steps within each episode

            # Run the episode until all values are imputed
            while not done:
                position = random.choice(self.env.missing_indices)
                action = self.choose_action(position)
                next_state, reward, done = self.env.step(action, position)
                self.learn(position, action, reward, next_state)
                state = next_state
                step_count += 1

            steps_per_episode.append(step_count)  # Record steps for this episode

            # Calculate MAE and RMSE for this episode
            mae, rmse = calculate_metrics(self.env, self)
            mae_per_episode.append(mae)
            rmse_per_episode.append(rmse)

            logging.info(f"Episode {episode} completed, steps={step_count}, epsilon={self.epsilon:.4f}")
            logging.info(f"Episode {episode} - MAE = {mae:.6f}, RMSE = {rmse:.6f}")

            # Decay epsilon after each episode
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Final metrics
        final_episode_steps = steps_per_episode[-1] if steps_per_episode else 0
        average_steps = sum(steps_per_episode) / len(steps_per_episode) if steps_per_episode else 0

        return mae, rmse, steps_per_episode, mae_per_episode, rmse_per_episode, final_episode_steps, average_steps


class ImputationEnvironment:
    def __init__(self, incomplete_data, complete_data, missing_rate):
        self.incomplete_data = incomplete_data.copy()
        self.complete_data = complete_data
        self.state = incomplete_data.copy()
        self.missing_indices = np.argwhere(pd.isna(self.incomplete_data.values))
        self.missing_rate = missing_rate
    def reset(self):
        self.state = self.incomplete_data.copy()
        return self.state

    def step(self, action, position):
        row, col = position
        self.state.iat[row, col] = action
        reward = -abs(self.complete_data.iat[row, col] - action)
        done = not pd.isna(self.state.values).any()
        return self.state, reward, done

    def get_possible_actions(self, col):
        """Return possible actions (values) for a column (excluding NaN)."""
        return self.complete_data.iloc[:, col].dropna().unique()


# Function to split dataset into training and test sets
def split_dataset(complete_data, missing_rate, test_size=0.3, random_state=42):
    # Perform the train-test split on complete_data
    complete_data_train, complete_data_test = train_test_split(
        complete_data, test_size=test_size, random_state=random_state)

    # Generate missing values for only the training set
    incomplete_data_train = generate_missing_df(complete_data_train, missing_rate)

    # Return both the training set (with missing data) and test set (complete)
    return complete_data_train, incomplete_data_train, complete_data_test, complete_data_test  # Unaltered test set



def save_training_results(dataset_id, missing_rate, results_dir, steps_per_episode, mae_per_episode, rmse_per_episode,
                          final_episode_steps, average_steps, imputed_data=None):
    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Path for the CSV file that will store episode metrics
    metrics_file_path = os.path.join(results_dir, f"{dataset_id}_missing_rate_{int(missing_rate * 100)}_metrics.csv")

    # Save steps, MAE, and RMSE per episode to the CSV file
    with open(metrics_file_path, mode='w', newline='') as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(["Episode", "Steps", "MAE", "RMSE"])
        for ep, steps, mae, rmse in zip(range(1, len(steps_per_episode) + 1), steps_per_episode, mae_per_episode,
                                        rmse_per_episode):
            writer.writerow([ep, steps, mae, rmse])
        writer.writerow(["Final Episode Steps", final_episode_steps])
        writer.writerow(["Average Steps", average_steps])
    logging.info(f"Metrics data saved to {metrics_file_path}")

    # Save final imputed data if provided
    if imputed_data is not None:
        file_name = f"{dataset_id}_missing_rate_{int(missing_rate * 100)}.csv"
        file_path = os.path.join(results_dir, file_name)
        imputed_data.to_csv(file_path, index=False)
        logging.info(f"Final imputed data saved to {file_path}")


def run_experiment(dataset_id, missing_rate):
    logging.info(f"Processing dataset ID: {dataset_id} with missing rate: {missing_rate}")

    # Load and split dataset
    complete_data, incomplete_data = load_dataset(dataset_id, missing_rate)
    complete_data_train, incomplete_data_train, complete_data_test, _ = split_dataset(complete_data, missing_rate)

    # Set up training environment
    env = ImputationEnvironment(incomplete_data_train, complete_data_train, missing_rate)
    agent = RLImputer(env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)

    # Run training
    final_mae, final_rmse, steps_per_episode, mae_per_episode, rmse_per_episode, final_episode_steps, average_steps = agent.train(
        episodes=250)

    # Save the results
    save_training_results(
        dataset_id=dataset_id,
        missing_rate=missing_rate,
        results_dir="./results",
        steps_per_episode=steps_per_episode,
        mae_per_episode=mae_per_episode,
        rmse_per_episode=rmse_per_episode,
        final_episode_steps=final_episode_steps,
        average_steps=average_steps,
        imputed_data=env.state
    )

    logging.info(f"Completed processing for dataset ID: {dataset_id} with missing rate: {missing_rate}")




if __name__ == "__main__":
    # dataset_ids = [94, 59, 17, 332, 350, 189, 484, 149]  # all datasets
    dataset_ids = [94]  # Define datasets for testing
    missing_rates = [0.05, 0.10, 0.15]  # Define missing rates

    # Create a list of all experiments (dataset_id, missing_rate)
    experiments = [(dataset_id, missing_rate) for dataset_id in dataset_ids for missing_rate in missing_rates]

    # Run experiments in parallel using multiprocessing
    with mp.Pool(processes=12) as pool:  # Adjust 'processes' based on available CPU
        pool.starmap(run_experiment, experiments)

    logging.info("All experiments completed.")
