import logging
import random
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import multiprocessing as mp
import csv

from ucimlrepo import fetch_ucirepo
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler(log_filename, mode='w')  # Output to file
    ]
)
import json

def save_json_results(dataset_id, missing_rate, test_mae, test_rmse, mae_rmse_results, results_dir="./results"):
    os.makedirs(results_dir, exist_ok=True)
    dataset_name = dataset_mapping.get(dataset_id, f"dataset_{dataset_id}")
    missing_rate_percent = int(missing_rate * 100)

    # Prepare the result entry
    result_entry = {
        "Dataset ID": dataset_id,
        "Dataset Name": dataset_name,
        "Missing Rate": missing_rate_percent,
        "Test MAE": test_mae,
        "Test RMSE": test_rmse,
        "Better than Benchmark": False
    }

    # Get benchmark values
    benchmark = mae_rmse_results.get(dataset_name, {}).get(missing_rate_percent, {})
    if benchmark:
        benchmark_mae = benchmark.get("MAE", float('inf'))
        benchmark_rmse = benchmark.get("RMSE", float('inf'))

        # Determine if our results are better
        if test_mae < benchmark_mae and test_rmse < benchmark_rmse:
            result_entry["Better than Benchmark"] = True
    else:
        benchmark_mae = None
        benchmark_rmse = None

    result_entry["Benchmark MAE"] = benchmark_mae
    result_entry["Benchmark RMSE"] = benchmark_rmse

    # Load existing results
    json_file = os.path.join(results_dir, "results_summary.json")
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            results_list = json.load(f)
    else:
        results_list = []

    # Update or append the new result
    # Remove any existing entry for this dataset and missing rate
    results_list = [entry for entry in results_list if not (entry["Dataset ID"] == dataset_id and entry["Missing Rate"] == missing_rate_percent)]
    results_list.append(result_entry)

    # Save back to JSON file
    with open(json_file, 'w') as f:
        json.dump(results_list, f, indent=4)

    logging.info(f"JSON results saved for dataset {dataset_id} with missing rate {missing_rate_percent}%.")

def calculate_metrics(env):
    # Get the imputed data (after training)
    imputed_data = env.state

    # Check for NaN values in complete_data and imputed_data
    if env.complete_data.isna().sum().sum() > 0:
        nan_counts = env.complete_data.isna().sum()  # Series with NaN counts per column
        nan_columns = nan_counts[nan_counts > 0]  # Filter columns with NaNs
        logging.error("NaN values found in complete_data:")
        for col, count in nan_columns.items():
            logging.error(f"Complete Data: Column '{col}': {count} NaN values")
        raise Exception("env.complete_data.")  # Early exit if complete_data has NaNs

    if imputed_data.isna().sum().sum() > 0:
        nan_counts = imputed_data.isna().sum()  # Series with NaN counts per column
        nan_columns = nan_counts[nan_counts > 0]  # Filter columns with NaNs
        logging.error("NaN values found in complete_data:")
        for col, count in nan_columns.items():
            logging.error(f"Imputed Data: Column '{col}': {count} NaN values")
        raise Exception("env.imputed_data.")  # Early exit if complete_data has NaNs

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

    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

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

    # Generate missing values for incomplete_data using the now fully populated complete_data
    incomplete_data = generate_missing_df(complete_data.copy(), missing_rate)  # Introduce experimental missing values

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

    def apply_policy(self, env):
        """Apply the trained Q-table to impute missing values in the test environment."""
        for position in env.missing_indices:
            state_key = tuple(position)
            # Choose the best action based on learned Q-values
            actions = env.get_possible_actions(position[1])
            if actions.size > 0:  # Check if actions array is not empty
                q_values = {a: self.q_table[state_key][a] for a in actions}
                best_action = max(q_values, key=q_values.get)
                env.state.iat[position[0], position[1]] = best_action
        return env.state

    def learn(self, position, action, reward, next_position):
        """Update the Q-table based on the action taken."""
        state_key = tuple(position)
        next_state_key = tuple(next_position)

        q_predict = self.q_table[state_key][action]
        q_target = reward + self.gamma * max(self.q_table[next_state_key].values(), default=0)
        self.q_table[state_key][action] += self.alpha * (q_target - q_predict)

    def train(self, datasetid, episodes, missing_rate, test_env, test_interval=50):
        steps_per_episode = []  # Track steps for each episode
        epsilon_per_episode = []  # Track epsilon values for each episode
        train_mae_per_episode, train_rmse_per_episode = [], []
        test_mae_per_interval, test_rmse_per_interval = [], []

        for episode in range(1, episodes + 1):
            state = self.env.reset()
            done = False
            step_count = 0

            # Run the episode until all values are imputed
            while not done:
                position = random.choice(self.env.missing_indices)
                action = self.choose_action(position)
                next_state, reward, done = self.env.step(action, position)
                self.learn(position, action, reward, next_state)
                state = next_state
                step_count += 1

            steps_per_episode.append(step_count)  # Record steps for this episode
            epsilon_per_episode.append(self.epsilon)  # Record epsilon for this episode

            # Calculate MAE and RMSE on the training set
            train_mae, train_rmse = calculate_metrics(self.env)
            train_mae_per_episode.append(train_mae)
            train_rmse_per_episode.append(train_rmse)
            logging.info(
                f"Dataset {datasetid} with missing rate {missing_rate}: Episode {episode} - Training MAE = {train_mae:.6f}, RMSE = {train_rmse:.6f}, Epsilon = {self.epsilon:.4f}")

            # Periodic testing on the test set
            if episode % test_interval == 0:
                # Reset test environment
                test_env.reset()

                # Apply current policy to test environment
                self.apply_policy(test_env)

                # Calculate MAE and RMSE on the test set
                test_mae, test_rmse = calculate_metrics(test_env)
                test_mae_per_interval.append((episode, test_mae))
                test_rmse_per_interval.append((episode, test_rmse))
                logging.info(
                    f"Dataset {datasetid} with missing rate {missing_rate}: Episode {episode} - Test MAE = {test_mae:.6f}, RMSE = {test_rmse:.6f}")

            # Decay epsilon after each episode
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        final_steps = steps_per_episode[-1] if steps_per_episode else 0
        avg_steps = sum(steps_per_episode) / len(steps_per_episode) if steps_per_episode else 0

        logging.info(f"Training completed for dataset {datasetid} with missing rate {missing_rate}.")
        return (train_mae_per_episode, train_rmse_per_episode, final_steps, avg_steps, steps_per_episode,
                epsilon_per_episode, test_mae_per_interval, test_rmse_per_interval)


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
    incomplete_data_train = generate_missing_df(complete_data_train.copy(), missing_rate)

    # Generate missing values for the test set (optional)
    incomplete_data_test = generate_missing_df(complete_data_test.copy(), missing_rate)

    # Return the complete and incomplete versions of both training and test sets
    return complete_data_train, incomplete_data_train, complete_data_test, incomplete_data_test


def save_training_results(dataset_id, missing_rate, results_dir, steps_per_episode, epsilon_per_episode,
                          mae_per_episode, rmse_per_episode, final_episode_steps, average_steps, imputed_data=None,
                          test_mae_intervals=None, test_rmse_intervals=None):
    os.makedirs(results_dir, exist_ok=True)
    metrics_file_path = os.path.join(results_dir, f"{dataset_id}_missing_rate_{int(missing_rate * 100)}_metrics.csv")

    with open(metrics_file_path, mode='w', newline='') as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(["Episode", "Steps", "Epsilon", "Training MAE", "Training RMSE", "Test MAE", "Test RMSE"])

        max_length = max(len(steps_per_episode), len(mae_per_episode), len(test_mae_intervals or []))
        test_interval_dict = dict(test_mae_intervals) if test_mae_intervals else {}
        test_rmse_interval_dict = dict(test_rmse_intervals) if test_rmse_intervals else {}

        for ep in range(1, len(steps_per_episode) + 1):
            steps = steps_per_episode[ep - 1]
            eps = epsilon_per_episode[ep - 1]
            mae = mae_per_episode[ep - 1]
            rmse = rmse_per_episode[ep - 1]
            test_mae = test_interval_dict.get(ep, "")
            test_rmse = test_rmse_interval_dict.get(ep, "")
            writer.writerow([ep, steps, eps, mae, rmse, test_mae, test_rmse])

        writer.writerow([])  # empty row
        writer.writerow(["Final Episode Steps", final_episode_steps])
        writer.writerow(["Average Steps", average_steps])

        if test_mae_intervals and test_rmse_intervals:
            final_test_mae = test_mae_intervals[-1][1]
            final_test_rmse = test_rmse_intervals[-1][1]
            writer.writerow(["Final Test MAE", final_test_mae])
            writer.writerow(["Final Test RMSE", final_test_rmse])

    if imputed_data is not None:
        file_name = f"{dataset_id}_missing_rate_{int(missing_rate * 100)}.csv"
        file_path = os.path.join(results_dir, file_name)
        imputed_data.to_csv(file_path, index=False)





def run_experiment(dataset_id, missing_rate):
    logging.info(f"Processing dataset ID: {dataset_id} with missing rate: {missing_rate}")

    # Load and split dataset
    complete_data, incomplete_data = load_dataset(dataset_id, missing_rate)
    complete_data_train, incomplete_data_train, complete_data_test, incomplete_data_test = split_dataset(complete_data, missing_rate)

    # Set up training environment
    env = ImputationEnvironment(incomplete_data_train, complete_data_train, missing_rate)
    agent = RLImputer(env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.990, epsilon_min=0.01)

    # Set up test environment
    test_env = ImputationEnvironment(incomplete_data=incomplete_data_test, complete_data=complete_data_test, missing_rate=missing_rate)

    # Train the agent with periodic testing
    (train_mae_per_episode, train_rmse_per_episode, final_steps, avg_steps, steps_per_episode,
     epsilon_per_episode, test_mae_per_interval, test_rmse_per_interval) = agent.train(
        dataset_id, episodes=600, missing_rate=missing_rate, test_env=test_env, test_interval=50)

    # Apply trained policy to test environment and calculate final test metrics
    test_env.reset()  # Reset test environment before final testing
    agent.apply_policy(test_env)
    test_mae, test_rmse = calculate_metrics(test_env)

    # Save or log results
    save_training_results(
        dataset_id=dataset_id,
        missing_rate=missing_rate,
        results_dir="./results",
        steps_per_episode=steps_per_episode,
        epsilon_per_episode=epsilon_per_episode,
        mae_per_episode=train_mae_per_episode,
        rmse_per_episode=train_rmse_per_episode,
        final_episode_steps=final_steps,
        average_steps=avg_steps,
        imputed_data=test_env.state,
        test_mae_intervals=test_mae_per_interval,
        test_rmse_intervals=test_rmse_per_interval
    )

    # Save results to JSON for comparison
    save_json_results(
        dataset_id=dataset_id,
        missing_rate=missing_rate,
        test_mae=test_mae,
        test_rmse=test_rmse,
        mae_rmse_results=mae_rmse_results,
        results_dir="./results"
    )



if __name__ == "__main__":
    #dataset_ids = [94, 59, 17, 332, 350, 189, 484, 149]  # all datasets
    dataset_ids = [17]  # all datasets
    #missing_rates = [0.05, 0.10, 0.15, 0.20]  # missing rates
    missing_rates = [0.05]

    # Create a list of all experiments (dataset_id, missing_rate)
    experiments = [(dataset_id, missing_rate) for dataset_id in dataset_ids for missing_rate in missing_rates]

    # Run experiments in parallel using multiprocessing
    with mp.Pool(processes=3) as pool:  # Adjust 'processes' based on available CPU
        pool.starmap(run_experiment, experiments)

    logging.info("All experiments completed.")
