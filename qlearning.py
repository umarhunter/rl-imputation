import logging
import random
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import csv
import itertools
import json

from collections import defaultdict
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Dataset mapping: IDs to Names
dataset_mapping = {
    94: "spambase",
    59: "letter_recognition",
    17: "breast_cancer_wisconsin",
    332: "online_news_popularity",
    350: "default_credit_card_clients",
    189: "parkinsons_telemonitoring",
    484: "travel_reviews",
    149: "statlog_vehicle_silhouettes"
}

mae_rmse_results = {
    "spambase": {
        5: {"MAE": 0.0183, "RMSE": 0.0485},
        10: {"MAE": 0.0191, "RMSE": 0.0494},
        15: {"MAE": 0.0194, "RMSE": 0.0511},
        20: {"MAE": 0.0198, "RMSE": 0.0527},
    },
    "letter_recognition": {
        5: {"MAE": 0.0826, "RMSE": 0.1183},
        10: {"MAE": 0.0916, "RMSE": 0.1251},
        15: {"MAE": 0.0986, "RMSE": 0.1301},
        20: {"MAE": 0.1004, "RMSE": 0.1373},
    },
    "default_credit_card": {
        5: {"MAE": 0.0212, "RMSE": 0.0494},
        10: {"MAE": 0.0223, "RMSE": 0.0501},
        15: {"MAE": 0.0233, "RMSE": 0.0527},
        20: {"MAE": 0.0289, "RMSE": 0.0536},
    },
    "news_popularity": {
        5: {"MAE": 0.0334, "RMSE": 0.0866},
        10: {"MAE": 0.0336, "RMSE": 0.0889},
        15: {"MAE": 0.0348, "RMSE": 0.0961},
        20: {"MAE": 0.0381, "RMSE": 0.1016},
    },
    "statlog_vehicle_silhouettes": {
        5: {"MAE": 0.1181, "RMSE": 0.1428},
        10: {"MAE": 0.1226, "RMSE": 0.1877},
        15: {"MAE": 0.1362, "RMSE": 0.1991},
        20: {"MAE": 0.1476, "RMSE": 0.2063},
    },
    "parkinsons_telemonitoring": {
        5: {"MAE": 0.0646, "RMSE": 0.1074},
        10: {"MAE": 0.0703, "RMSE": 0.1148},
        15: {"MAE": 0.0754, "RMSE": 0.1221},
        20: {"MAE": 0.0797, "RMSE": 0.1286},
    },
    "breast_cancer_wisconsin": {
        5: {"MAE": 0.0796, "RMSE": 0.1152},
        10: {"MAE": 0.0814, "RMSE": 0.1183},
        15: {"MAE": 0.0847, "RMSE": 0.1236},
        20: {"MAE": 0.0847, "RMSE": 0.1236},
    },
    "travel_reviews": {
        5: {"MAE": 0.0646, "RMSE": 0.1128},
        10: {"MAE": 0.0824, "RMSE": 0.1373},
        15: {"MAE": 0.0893, "RMSE": 0.1424},
        20: {"MAE": 0.0893, "RMSE": 0.1424},
    },
}


# Configure logging to write to both console and file
log_filename = "training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',  # Include processName
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(log_filename, mode='w')  # Output to file
    ]
)

def calculate_metrics(env, scaler):
    """
    Calculate MAE and RMSE between true and imputed values after inverse scaling.

    Parameters:
    - env (ImputationEnvironment): The environment containing the complete and imputed data.
    - scaler (MinMaxScaler): The scaler used for inverse transformations.

    Returns:
    - mae (float): Mean Absolute Error.
    - rmse (float): Root Mean Squared Error.
    """
    # Get missing positions
    missing_positions = env.missing_indices  # array of (row, col)

    # Extract true and imputed values at missing positions
    true_values = env.complete_data.values[missing_positions[:, 0], missing_positions[:, 1]]
    imputed_values = env.state.values[missing_positions[:, 0], missing_positions[:, 1]]

    # Handle any remaining NaN values in imputed data (if any)
    nan_indices = np.isnan(imputed_values)
    if np.any(nan_indices):
        imputed_values[nan_indices] = env.initial_estimate.values[missing_positions[:, 0], missing_positions[:, 1]][nan_indices]

    # Get column indices
    cols = env.complete_data.columns[missing_positions[:, 1]]

    # Prepare arrays for inverse transformation
    true_inv = np.empty_like(true_values)
    imputed_inv = np.empty_like(imputed_values)

    for i, col in enumerate(cols):
        # Get the scaler parameters for the current column
        col_idx = env.complete_data.columns.get_loc(col)
        min_val = scaler.min_[col_idx]
        scale = scaler.scale_[col_idx]

        # Inverse transform the true and imputed values
        true_inv[i] = true_values[i] * scale + min_val
        imputed_inv[i] = imputed_values[i] * scale + min_val

    # Calculate MAE and RMSE
    mae = mean_absolute_error(true_inv, imputed_inv)
    rmse = root_mean_squared_error(true_inv, imputed_inv)

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
    for row_idx, col_idx in zip(*multi_dim_indices): # Unpack the multi-dimensional indices
        df_with_missing.iat[row_idx, col_idx] = np.nan

    return df_with_missing

def load_dataset(datasetid):
    """
    Load and preprocess the dataset.

    Parameters:
    - datasetid (int): The ID of the dataset to load.

    Returns:
    - complete_data (pd.DataFrame): Preprocessed complete dataset with missing values imputed by column means.
    """
    dataset_name = dataset_mapping.get(datasetid, f"dataset_{datasetid}")
    cache_file = f"./data/{dataset_name}.csv"

    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        metadata_file = f"./data/{dataset_name}_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        target_columns = metadata.get('target_columns', [])
    else:
        dataset = fetch_ucirepo(id=datasetid)
        df = dataset.data.original
        target_columns = dataset.metadata.target_col
        metadata = {'target_columns': target_columns}
        os.makedirs('./data', exist_ok=True)
        df.to_csv(cache_file, index=False)
        with open(f"./data/{dataset_name}_metadata.json", 'w') as f:
            json.dump(metadata, f)

    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

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

    return complete_data


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
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.05):
        logging.info("Initializing Q-Learning Agent")
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.actions = [-1, 1]
        self.q_table = defaultdict(lambda: defaultdict(float))

    def choose_action(self, state):
        """Choose an action using an epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            if q_values:
                action = max(q_values, key=q_values.get)
            else:
                # If state is unseen, choose a random action
                action = random.choice(self.actions)
        return action

    def learn(self, state, action, reward, next_state):
        """Update the Q-table based on the action taken."""
        q_predict = self.q_table[state][action]
        q_target = reward + self.gamma * max(self.q_table[next_state].values(), default=0)
        self.q_table[state][action] += self.alpha * (q_target - q_predict)

    def train(self, dataset_name, episodes, missing_rate, test_env, scaler, test_interval=50, results_dir="./results"):
        # Initialize metrics
        train_metrics = {
            "episodes": [],
            "steps_per_episode": [],
            "epsilon_per_episode": [],
            "mae_per_episode": [],
            "rmse_per_episode": [],
            "average_reward_per_episode": []
        }
        test_metrics = {
            "test_mae_per_interval": [],
            "test_rmse_per_interval": []
        }

        # Prepare directories and files for real-time logging
        os.makedirs(results_dir, exist_ok=True)
        missing_rate_percent = int(missing_rate * 100)

        # Paths for saving metrics
        train_metrics_file = os.path.join(results_dir, 
            f"{dataset_name}_missing_rate_{missing_rate_percent}_train_metrics.csv")
        test_metrics_file = os.path.join(results_dir, 
            f"{dataset_name}_missing_rate_{missing_rate_percent}_test_metrics.csv")

        # Initialize CSV writers
        with open(train_metrics_file, mode='w', newline='') as train_file, \
             open(test_metrics_file, mode='w', newline='') as test_file:

            train_writer = csv.writer(train_file)
            train_writer.writerow(["Episode", "Steps", "Epsilon", "Training MAE", "Training RMSE", "Average Reward"])

            test_writer = csv.writer(test_file)
            test_writer.writerow(["Episode", "Test MAE", "Test RMSE"])

            for episode in range(1, episodes + 1):
                self.env.reset()
                step_count = 0
                total_reward = 0

                # For each missing value
                for position in self.env.missing_indices:
                    done = False
                    position_steps = 0
                    max_position_steps = 10000

                    while not done and position_steps < max_position_steps:
                        # Get current state from the environment
                        state = self.env.get_state(position)
                        action = self.choose_action(state)
                        # Apply action and get reward from the environment
                        _, reward, done = self.env.step(action, position)
                        # Get next state from the environment
                        next_state = self.env.get_state(position)
                        # Learn from the experience
                        self.learn(state, action, reward, next_state)
                        position_steps += 1
                        step_count += 1
                        total_reward += reward

                # Decay epsilon after each episode
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                # Calculate MAE and RMSE on the training set
                train_mae, train_rmse = calculate_metrics(self.env, scaler)

                # Calculate average reward
                average_reward = total_reward / len(self.env.missing_indices) if len(self.env.missing_indices) > 0 else 0

                # Append metrics
                train_metrics["episodes"].append(episode)
                train_metrics["steps_per_episode"].append(step_count)
                train_metrics["epsilon_per_episode"].append(self.epsilon)
                train_metrics["mae_per_episode"].append(train_mae)
                train_metrics["rmse_per_episode"].append(train_rmse)
                train_metrics["average_reward_per_episode"].append(average_reward)

                # Write training metrics to CSV
                train_writer.writerow([episode, step_count, self.epsilon, train_mae, train_rmse, average_reward])

                logging.info(
                    f"Dataset {dataset_name} MR {missing_rate}: Episode {episode} - "
                    f"Training MAE = {train_mae:.6f}, RMSE = {train_rmse:.6f}, "
                    f"Epsilon = {self.epsilon:.4f}, Steps = {step_count}, Avg Reward = {average_reward:.2f}"
                )

                # Periodic testing on the test set
                if episode % test_interval == 0:
                    test_env.reset()
                    self.apply_policy(test_env)
                    test_mae, test_rmse = calculate_metrics(test_env, scaler)
                    test_metrics["test_mae_per_interval"].append((episode, test_mae))
                    test_metrics["test_rmse_per_interval"].append((episode, test_rmse))

                    # Write test metrics to CSV
                    test_writer.writerow([episode, test_mae, test_rmse])

                    logging.info(
                        f"Dataset {dataset_name} MR {missing_rate} Episode {episode} - "
                        f"Test MAE = {test_mae:.6f}, RMSE = {test_rmse:.6f}"
                    )
                
                # Periodically log Q-table samples
                if episode % 100 == 0:
                    sample_states = list(self.q_table.keys())[:5]  # Sample first 5 states
                    sample_q = {state: dict(q_values) for state, q_values in self.q_table.items() if state in sample_states}
                    logging.info(f"Episode {episode} Q-table sample: {sample_q}")

        # Log the lowest test MAE and RMSE
        if test_metrics["test_mae_per_interval"]:
            lowest_test_mae = min(mae for _, mae in test_metrics["test_mae_per_interval"])
            lowest_test_rmse = min(rmse for _, rmse in test_metrics["test_rmse_per_interval"])
            logging.info(
                f"Dataset {dataset_name} MR {missing_rate}: Lowest Test MAE = {lowest_test_mae:.6f}, "
                f"Lowest Test RMSE = {lowest_test_rmse:.6f}"
            )
        else:
            logging.warning(f"No test metrics recorded for dataset {dataset_name} MR {missing_rate}.")

        logging.info(f"Training completed for dataset {dataset_name} with missing rate {missing_rate}.")
        return train_metrics, test_metrics

    def apply_policy(self, env):
        """Apply the trained Q-table to impute missing values in the environment."""
        for position in env.missing_indices:
            done = False
            position_steps = 0
            max_position_steps = 10000
            while not done and position_steps < max_position_steps:
                state = env.get_state(position)
                q_values = self.q_table[state]
                if q_values:
                    action = max(q_values, key=q_values.get)
                else:
                    action = random.choice(self.actions)  # If state unseen, choose random action
                _, _, done = env.step(action, position)
                position_steps += 1




class ImputationEnvironment:
    def __init__(self, incomplete_data, complete_data, missing_rate, adjustment_factor=0.01, error_threshold=0.01):
        self.incomplete_data = incomplete_data.copy()
        self.complete_data = complete_data
        self.state = incomplete_data.copy()
        self.missing_indices = np.argwhere(pd.isna(self.incomplete_data.values))
        self.missing_rate = missing_rate
        
        # Initial estimate using mean imputation
        self.initial_estimate = incomplete_data.fillna(incomplete_data.mean())

        self.adjustment_factor = adjustment_factor
        self.error_threshold = error_threshold

        # Compute maximum possible error per column for normalization
        self.max_error_per_column = (self.complete_data.max() - self.complete_data.min()).to_dict()

        # Number of states (as per the paper)
        self.n_states = 30

        # Initialize the R-matrix
        self.R_matrix = self.initialize_r_matrix()

    def initialize_r_matrix(self):
        # Initialize R-matrix with -1
        R_matrix = np.full((self.n_states, self.n_states), -1)

        # Set rewards for viable transitions
        for i in range(self.n_states):
            if i > 0:
                R_matrix[i, i - 1] = 0  # Moving to a lower error state
            if i < self.n_states - 1:
                R_matrix[i, i + 1] = 0  # Moving to a higher error state

        # Set high reward for transitions to the goal state (state 0)
        for i in range(1, self.n_states):
            R_matrix[i, 0] = 100

        return R_matrix
    

    def get_state(self, position):
        row, col = position
        current_value = self.state.iat[row, col]

        if np.isnan(current_value):
            current_value = self.initial_estimate.iat[row, col]

        true_value = self.complete_data.iat[row, col]
        error = abs(true_value - current_value)

        # Get the column name from the column index
        col_name = self.state.columns[col]

        # Normalize error between 0 and 1 using the column's max error
        max_error = self.max_error_per_column[col_name]
        if max_error == 0:
            normalized_error = 0
        else:
            normalized_error = error / max_error
            normalized_error = min(normalized_error, 1)  # Cap at 1

        # Discretize normalized error into bins (states)
        error_bin = int(normalized_error * (self.n_states - 1))

        state = (col_name, error_bin)
        return state

    def get_state_with_value(self, position, value):
        row, col = position

        true_value = self.complete_data.iat[row, col]
        error = abs(true_value - value)

        # Get the column name from the column index
        col_name = self.state.columns[col]

        # Normalize error between 0 and 1 using the column's max error
        max_error = self.max_error_per_column[col_name]
        if max_error == 0:
            normalized_error = 0
        else:
            normalized_error = error / max_error
            normalized_error = min(normalized_error, 1)  # Cap at 1

        # Discretize normalized error into bins (states)
        error_bin = int(normalized_error * (self.n_states - 1))

        state = (col_name, error_bin)
        return state

    def reset(self):
        self.state = self.incomplete_data.copy()
        return self.state

    def step(self, action, position):
        row, col = position
        current_value = self.state.iat[row, col]

        if np.isnan(current_value):
            current_value = self.initial_estimate.iat[row, col]
            # Do not modify self.state here

        # Get the current state before applying the action
        current_state = self.get_state_with_value(position, current_value)

        # Apply action using multiplicative adjustment
        new_value = current_value * (1 + self.adjustment_factor * action)

        # Get the column name from the column index
        col_name = self.state.columns[col]
        col_min = self.complete_data[col_name].min()
        col_max = self.complete_data[col_name].max()

        # Clip the new value to prevent it from going out of bounds
        new_value = np.clip(new_value, col_min, col_max)
        self.state.iat[row, col] = new_value

        # Get the next state after applying the action
        next_state = self.get_state(position)  # Now self.state has been updated

        # Get reward from R-matrix
        current_state_index = current_state[1]
        next_state_index = next_state[1]
        reward = self.R_matrix[current_state_index, next_state_index]

        # Check if goal state (state 0) is reached
        done = next_state_index == 0

        return self.state, reward, done


    def get_possible_actions(self, col):
        """Return possible actions (values) for a column (excluding NaN)."""
        return self.complete_data.iloc[:, col].dropna().unique()


def split_dataset(complete_data, missing_rate, test_size=0.3, random_state=42):
    """
    Split the dataset into training and testing sets, scale them, and introduce missing values.

    Parameters:
    - complete_data (pd.DataFrame): The complete dataset with missing values imputed by column means.
    - missing_rate (float): The proportion of data to set as missing (e.g., 0.05 for 5%).
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    - complete_data_train_scaled (pd.DataFrame): Scaled training complete data.
    - incomplete_data_train (pd.DataFrame): Scaled training incomplete data with missing values.
    - complete_data_test_scaled (pd.DataFrame): Scaled testing complete data.
    - incomplete_data_test (pd.DataFrame): Scaled testing incomplete data with missing values.
    - scaler (MinMaxScaler): The fitted scaler used for inverse transformations.
    """
    # Split the dataset into training and testing sets
    complete_data_train, complete_data_test = train_test_split(
        complete_data, test_size=test_size, random_state=random_state
    )

    # Initialize the scaler and fit on the training data
    scaler = MinMaxScaler()
    complete_data_train_scaled = pd.DataFrame(
        scaler.fit_transform(complete_data_train), columns=complete_data_train.columns, index=complete_data_train.index
    )

    # Transform the testing data using the same scaler
    complete_data_test_scaled = pd.DataFrame(
        scaler.transform(complete_data_test), columns=complete_data_test.columns, index=complete_data_test.index
    )

    # Generate missing values on the scaled training data
    if missing_rate > 0:
        incomplete_data_train = generate_missing_df(complete_data_train_scaled.copy(), missing_rate)
    else:
        incomplete_data_train = complete_data_train_scaled.copy()

    # Generate missing values on the scaled testing data
    if missing_rate > 0:
        incomplete_data_test = generate_missing_df(complete_data_test_scaled.copy(), missing_rate)
    else:
        incomplete_data_test = complete_data_test_scaled.copy()

    return complete_data_train_scaled, incomplete_data_train, complete_data_test_scaled, incomplete_data_test, scaler



def save_training_results(dataset_name, missing_rate, results_dir, 
                          train_metrics, test_metrics, imputed_data=None):
    """
    Save training and test metrics to CSV files and optionally save imputed data.
    
    Parameters:
    - dataset_name (str): Name of the dataset.
    - missing_rate (float): Missing rate used in the experiment.
    - results_dir (str): Directory to save the results.
    - train_metrics (dict): Dictionary containing training metrics per episode.
    - test_metrics (dict): Dictionary containing test metrics per interval.
    - imputed_data (pd.DataFrame, optional): DataFrame containing the imputed data.
    """
    os.makedirs(results_dir, exist_ok=True)
    missing_rate_percent = int(missing_rate * 100)
    
    # Paths for saving metrics
    train_metrics_file = os.path.join(results_dir, 
        f"{dataset_name}_missing_rate_{missing_rate_percent}_train_metrics.csv")
    test_metrics_file = os.path.join(results_dir, 
        f"{dataset_name}_missing_rate_{missing_rate_percent}_test_metrics.csv")
    
    # Save training metrics
    with open(train_metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Steps", "Epsilon", "Training MAE", "Training RMSE"])
        for ep in range(len(train_metrics["steps_per_episode"])):
            writer.writerow([
                ep + 1,
                train_metrics["steps_per_episode"][ep],
                train_metrics["epsilon_per_episode"][ep],
                train_metrics["mae_per_episode"][ep],
                train_metrics["rmse_per_episode"][ep]
            ])
    
    # Save test metrics
    with open(test_metrics_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Test MAE", "Test RMSE"])
        for ep, mae in test_metrics["test_mae_per_interval"]:
            rmse = dict(test_metrics["test_rmse_per_interval"]).get(ep, "")
            writer.writerow([ep, mae, rmse])
    
    # # Optionally save the imputed data
    # if imputed_data is not None:
    #     imputed_data_file = os.path.join(results_dir, 
    #         f"{dataset_name}_missing_rate_{missing_rate_percent}_imputed_data.csv")
    #     imputed_data.to_csv(imputed_data_file, index=False)



def run_experiment(dataset_id, missing_rate):
    # Get the dataset name from the mapping
    dataset_name = dataset_mapping.get(dataset_id, f"dataset_{dataset_id}")
    logging.info(f"Processing dataset: {dataset_name} with missing rate: {missing_rate}")

    # Load and split dataset
    complete_data = load_dataset(dataset_id)
    complete_data_train, incomplete_data_train, complete_data_test, incomplete_data_test, scaler = split_dataset(complete_data, missing_rate)

    # Set up training environment
    env = ImputationEnvironment(incomplete_data_train, complete_data_train, missing_rate)
    agent = RLImputer(env, alpha=0.2, gamma=0.92, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1)

    # Set up test environment
    test_env = ImputationEnvironment(incomplete_data=incomplete_data_test, complete_data=complete_data_test, missing_rate=missing_rate)

    # Train the agent with early stopping
    train_metrics, test_metrics = agent.train(
        dataset_name, episodes=465, missing_rate=missing_rate, test_env=test_env, scaler=scaler, test_interval=5
    )

    # Apply trained policy to test environment and calculate final test metrics
    test_env.reset()  # Reset test environment before final testing
    agent.apply_policy(test_env)
    test_mae, test_rmse = calculate_metrics(test_env, scaler)

    # Save or log results
    save_training_results(
        dataset_name=dataset_name,
        missing_rate=missing_rate,
        results_dir="./results",
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        imputed_data=env.state
    )

def init_worker(complete_data_):
    global complete_data
    complete_data = complete_data_

def hyperparameter_tuning_single_combination(args):
    dataset_id, missing_rate, alpha, gamma = args
    logging.info(f"Testing alpha={alpha}, gamma={gamma}")

    # Set random seed for reproducibility
    seed = int(alpha * 1000 + gamma * 1000) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)

    try:
        # Load and split the dataset within each worker to avoid sharing large data across processes
        complete_data = load_dataset(dataset_id)
        complete_data_train_scaled, incomplete_data_train, complete_data_test_scaled, incomplete_data_test, scaler = split_dataset(
            complete_data, missing_rate
        )

        # Set up training environment
        env = ImputationEnvironment(incomplete_data_train, complete_data_train_scaled, missing_rate)
        agent = RLImputer(env, alpha=alpha, gamma=gamma, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1)

        # Set up test environment
        test_env = ImputationEnvironment(incomplete_data=incomplete_data_test, complete_data=complete_data_test_scaled, missing_rate=missing_rate)

        # Train the agent (reduced episodes for tuning)
        train_metrics, test_metrics = agent.train(
            dataset_name=dataset_mapping.get(dataset_id, f"dataset_{dataset_id}"),
            episodes=700,  # Reduced episodes for faster tuning
            missing_rate=missing_rate,
            test_env=test_env,
            scaler=scaler,
            test_interval=50,
        )

        # Apply trained policy to test environment and calculate final test metrics
        test_env.reset()
        agent.apply_policy(test_env)
        test_mae, test_rmse = calculate_metrics(test_env, scaler)

        logging.info(f"alpha={alpha}, gamma={gamma}, Test MAE={test_mae}")

        return (alpha, gamma, test_mae, test_rmse)
    except Exception as e:
        logging.exception(f"Error with alpha={alpha}, gamma={gamma}: {e}")
        return (alpha, gamma, float('inf'), float('inf'))




def prepare_hyperparameter_combinations(dataset_id, missing_rate):
    alpha_values = np.array([0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    gamma_values = np.arange(0.90, 1.00, 0.01)

    hyperparameter_combinations = [ 
        (dataset_id, missing_rate, alpha, gamma)
        for alpha, gamma in itertools.product(alpha_values, gamma_values)
    ]

    return hyperparameter_combinations

def hyperparameter_tuning_parallel(dataset_id, missing_rate):
    """
    Perform hyperparameter tuning in parallel for a given dataset and missing rate.

    Parameters:
    - dataset_id (int): ID of the dataset.
    - missing_rate (float): Missing rate used in the experiment.
    """
    combinations = prepare_hyperparameter_combinations(dataset_id, missing_rate)
    num_processes = min(14, mp.cpu_count())  # Adjust based on your system's capacity

    dataset_name = dataset_mapping.get(dataset_id, f"dataset_{dataset_id}")
    missing_rate_percent = int(missing_rate * 100)

    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{dataset_name}_mr_{missing_rate_percent}_hyperparameter_tuning_results.csv")

    best_mae = float('inf')
    best_result = None

    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Alpha", "Gamma", "Test MAE", "Test RMSE"])

        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(hyperparameter_tuning_single_combination, combinations)

        for res in results:
            writer.writerow(res)
            if res[2] < best_mae:
                best_mae = res[2]
                best_result = res

    if best_result:
        print(f"Best MAE: {best_mae} with parameters: alpha={best_result[0]}, gamma={best_result[1]}")
        logging.info(f"Best MAE: {best_mae} with parameters: alpha={best_result[0]}, gamma={best_result[1]}")
    else:
        print("No successful results were obtained.")
        logging.warning("No successful results were obtained.")






if __name__ == "__main__":
    dataset_ids = [17, 16]  # all datasets
    missing_rates = [0.05, 0.10, 0.15, 0.20]  # missing rates
    missing_rates = [0.05]  # missing rates
    # Create a list of all experiments (dataset_id, missing_rate)
    experiments = [(dataset_id, missing_rate) for dataset_id in dataset_ids for missing_rate in missing_rates]

    # for dataset_id in dataset_ids:
    #     # Load the complete dataset without missing values
    #     complete_data, _ = load_dataset(dataset_id, missing_rate=0)
    #     for missing_rate in missing_rates:
    #         hyperparameter_tuning_parallel(dataset_id, complete_data, missing_rate)

    try:
        with mp.Pool(processes=14) as pool:
            pool.starmap(run_experiment, experiments)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, terminating pool.")
        pool.terminate()  # Terminate all child processes immediately
        pool.join()  # Wait for the pool to finish cleanup
        print("All experiments were terminated.")

    logging.info("All experiments completed.")
