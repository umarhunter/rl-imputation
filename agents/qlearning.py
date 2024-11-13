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

def save_json_results(dataset_name, missing_rate, test_mae, test_rmse, mae_rmse_results, results_dir="./results"):
    os.makedirs(results_dir, exist_ok=True)
    missing_rate_percent = int(missing_rate * 100)

    # Prepare the result entry
    result_entry = {
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
    results_list = [entry for entry in results_list if not (entry["Dataset Name"] == dataset_name and entry["Missing Rate"] == missing_rate_percent)]
    results_list.append(result_entry)

    # Save back to JSON file
    with open(json_file, 'w') as f:
        json.dump(results_list, f, indent=4)

    logging.info(f"JSON results saved for dataset {dataset_name} with missing rate {missing_rate_percent}%.")


def calculate_metrics(env):
    # Get the imputed data (after training)
    imputed_data = env.state

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
    for row_idx, col_idx in zip(*multi_dim_indices): # Unpack the multi-dimensional indices
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
    def __init__(self, env, alpha=1.0, gamma=0.9, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.01):
        logging.info("Initializing Q-Learning Agent")
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.actions = [-1, 1]  # Decrease or increase actions
        self.q_table = defaultdict(lambda: defaultdict(float))

    def get_state(self, position, env):
        row, col = position
        current_value = env.state.iat[row, col]

        if np.isnan(current_value):
            current_value = env.initial_estimate.iat[row, col]

        true_value = env.complete_data.iat[row, col]
        error = abs(true_value - current_value)

        # Get the column name from the column index
        col_name = env.state.columns[col]
        # Get the standard deviation using the column name
        std_dev = env.column_std_dev[col_name]

        # Discretize the error into bins
        error_bin_size = std_dev * 0.1  # we can adjust

        if error_bin_size == 0:
            error_bin = 0
        else:
            error_bin = int(error / error_bin_size)

        state = (col_name, error_bin)
        return state

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

    def train(self, dataset_name, episodes, missing_rate, test_env, test_interval=50, patience=25):
        steps_per_episode = []
        epsilon_per_episode = []
        train_mae_per_episode, train_rmse_per_episode = [], []
        test_mae_per_interval, test_rmse_per_interval = [], []

        best_test_mae = float('inf')
        patience_counter = 0

        for episode in range(1, episodes + 1):
            self.env.reset()
            step_count = 0

            # For each missing value
            for position in self.env.missing_indices:
                done = False
                position_steps = 0
                max_position_steps = 200  # Adjusted as per the paper
                while not done and position_steps < max_position_steps:
                    state = self.get_state(position, self.env)
                    action = self.choose_action(state)
                    # Apply action and get reward
                    _, reward, done = self.env.step(action, position)
                    next_state = self.get_state(position, self.env)
                    # Learn from the experience
                    self.learn(state, action, reward, next_state)
                    position_steps += 1
                    step_count += 1

            steps_per_episode.append(step_count)
            epsilon_per_episode.append(self.epsilon)

            # Calculate MAE and RMSE on the training set
            train_mae, train_rmse = calculate_metrics(self.env)
            train_mae_per_episode.append(train_mae)
            train_rmse_per_episode.append(train_rmse)
            logging.info(
                f"Dataset {dataset_name} MR {missing_rate}: Episode {episode} - "
                f"Training MAE = {train_mae:.6f}, RMSE = {train_rmse:.6f}, "
                f"Epsilon = {self.epsilon:.4f}, Steps = {step_count}"
            )

            # Periodic testing on the test set
            if episode % test_interval == 0:
                test_env.reset()
                self.apply_policy(test_env)
                test_mae, test_rmse = calculate_metrics(test_env)
                test_mae_per_interval.append((episode, test_mae))
                test_rmse_per_interval.append((episode, test_rmse))
                logging.info(
                    f"Dataset {dataset_name} MR {missing_rate} Episode {episode} - "
                    f"Test MAE = {test_mae:.6f}, RMSE = {test_rmse:.6f}"
                )

                # Early stopping based on test MAE
                if test_mae < best_test_mae:
                    best_test_mae = test_mae
                    patience_counter = 0
                else:
                    patience_counter += test_interval
                    if patience_counter >= patience:
                        logging.info(f"Early stopping triggered after {episode} episodes.")
                        break  # Exit the training loop

            # Decay epsilon after each episode
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        final_steps = steps_per_episode[-1] if steps_per_episode else 0
        avg_steps = sum(steps_per_episode) / len(steps_per_episode) if steps_per_episode else 0

        logging.info(f"Training completed for dataset {dataset_name} with missing rate {missing_rate}.")
        return (train_mae_per_episode, train_rmse_per_episode, final_steps, avg_steps, steps_per_episode,
                epsilon_per_episode, test_mae_per_interval, test_rmse_per_interval)

    def apply_policy(self, env):
        """Apply the trained Q-table to impute missing values in the environment."""
        for position in env.missing_indices:
            done = False
            position_steps = 0
            max_position_steps = 200  # Same as in training
            while not done and position_steps < max_position_steps:
                state = self.get_state(position, env)
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
        # Compute standard deviation per column
        self.column_std_dev = self.complete_data.std()

    def reset(self):
        self.state = self.incomplete_data.copy()
        return self.state

    def step(self, action, position):
        row, col = position
        current_value = self.state.iat[row, col]

        if np.isnan(current_value):
            current_value = self.initial_estimate.iat[row, col]
            self.state.iat[row, col] = current_value

        true_value = self.complete_data.iat[row, col]

        # Calculate previous error before action
        previous_error = abs(true_value - current_value)

        # Apply action using multiplicative adjustment
        new_value = current_value * (1 + self.adjustment_factor * action)
        self.state.iat[row, col] = new_value

        # Calculate new error after action
        new_error = abs(true_value - new_value)

        # Calculate reward based on percentage improvement
        reward = previous_error - new_error

        # Check if error is below threshold
        done = new_error < self.error_threshold

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


def save_training_results(dataset_name, missing_rate, results_dir, steps_per_episode, epsilon_per_episode,
                          mae_per_episode, rmse_per_episode, final_episode_steps, average_steps, imputed_data=None,
                          test_mae_intervals=None, test_rmse_intervals=None):
    os.makedirs(results_dir, exist_ok=True)
    metrics_file_path = os.path.join(results_dir, f"{dataset_name}_missing_rate_{int(missing_rate * 100)}_metrics.csv")

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
        file_name = f"{dataset_name}_missing_rate_{int(missing_rate * 100)}.csv"
        file_path = os.path.join(results_dir, file_name)
        imputed_data.to_csv(file_path, index=False)



def run_experiment(dataset_id, missing_rate):
    # Get the dataset name from the mapping
    dataset_name = dataset_mapping.get(dataset_id, f"dataset_{dataset_id}")
    logging.info(f"Processing dataset: {dataset_name} with missing rate: {missing_rate}")

    # Load and split dataset
    complete_data, incomplete_data = load_dataset(dataset_id, missing_rate)
    complete_data_train, incomplete_data_train, complete_data_test, incomplete_data_test = split_dataset(complete_data, missing_rate)

    # Set up training environment
    env = ImputationEnvironment(incomplete_data_train, complete_data_train, missing_rate)
    agent = RLImputer(env, alpha=0.15, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1)

    # Set up test environment
    test_env = ImputationEnvironment(incomplete_data=incomplete_data_test, complete_data=complete_data_test, missing_rate=missing_rate)

    # Train the agent with early stopping
    (train_mae_per_episode, train_rmse_per_episode, final_steps, avg_steps, steps_per_episode,
     epsilon_per_episode, test_mae_per_interval, test_rmse_per_interval) = agent.train(
        dataset_name, episodes=10000, missing_rate=missing_rate, test_env=test_env, test_interval=1, patience=35)

    # Apply trained policy to test environment and calculate final test metrics
    test_env.reset()  # Reset test environment before final testing
    agent.apply_policy(test_env)
    test_mae, test_rmse = calculate_metrics(test_env)

    # Save or log results
    save_training_results(
        dataset_name=dataset_name,  # Updated parameter
        missing_rate=missing_rate,
        results_dir="./results",
        steps_per_episode=steps_per_episode,
        epsilon_per_episode=epsilon_per_episode,
        mae_per_episode=train_mae_per_episode,
        rmse_per_episode=train_rmse_per_episode,
        final_episode_steps=final_steps,
        average_steps=avg_steps,
        imputed_data=env.state,
        test_mae_intervals=test_mae_per_interval,
        test_rmse_intervals=test_rmse_per_interval
    )

    # Save results to JSON for comparison
    save_json_results(
        dataset_name=dataset_name,  # Updated parameter
        missing_rate=missing_rate,
        test_mae=test_mae,
        test_rmse=test_rmse,
        mae_rmse_results=mae_rmse_results,
        results_dir="./results"
    )




if __name__ == "__main__":
    dataset_ids = [17]  # all datasets
    missing_rates = [0.05, 0.10, 0.15, 0.20]  # missing rates

    # Create a list of all experiments (dataset_id, missing_rate)
    experiments = [(dataset_id, missing_rate) for dataset_id in dataset_ids for missing_rate in missing_rates]

    # Run experiments in parallel using multiprocessing
    try:
        with mp.Pool(processes=4) as pool:
            pool.starmap(run_experiment, experiments)
    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, terminating pool.")
        pool.terminate()  # Terminate all child processes immediately
        pool.join()  # Wait for the pool to finish cleanup
        print("All experiments were terminated.")

    logging.info("All experiments completed.")
