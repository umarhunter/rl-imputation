import logging
import random
import pickle
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

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

def save_checkpoint(agent, episode, imputed_data, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint = {
        'q_table': dict(agent.q_table),
        'episode': episode,
        'epsilon': agent.epsilon
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{episode}.pth")
    torch.save(checkpoint, checkpoint_path)
    imputed_data_path = os.path.join(checkpoint_dir, f"checkpoint_{episode}.csv")
    imputed_data.to_csv(imputed_data_path, index=False)
    logging.info(f"Checkpoint saved at episode {episode} with CSV file.")


def load_checkpoint(checkpoint_dir="checkpoints", checkpoint_file=None):
    if checkpoint_file is None:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if not checkpoint_files:
            return None
        checkpoint_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
        checkpoint_file = checkpoint_files[-1]

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    checkpoint = torch.load(checkpoint_path)
    logging.info(f"Checkpoint loaded from {checkpoint_file}")

    # Convert q_table back to defaultdict
    q_table = defaultdict(lambda: defaultdict(float), checkpoint['q_table'])
    return {
        'q_table': q_table,
        'episode': checkpoint['episode'],
        'epsilon': checkpoint['epsilon']
    }


def calculate_metrics(env, agent):
    # Get the imputed data (after training)
    imputed_data = env.state

    # Check for NaN values before calculating MAE and RMSE
    if imputed_data.isna().sum().sum() > 0:
            # Option 1: Return a failure metric or log it
        # return None, None
        
        # Option 2: Fill NaN values with mean (conservative fallback)
        imputed_data = imputed_data.fillna(imputed_data.mean())

    # Calculate MAE and RMSE between imputed data and complete data
    mae = mean_absolute_error(env.complete_data.values.flatten(), imputed_data.values.flatten())
    rmse = root_mean_squared_error(env.complete_data.values.flatten(), imputed_data.values.flatten())

    return mae, rmse


def get_toy_data():
    # Sample Data
    complete_data = pd.DataFrame({
        'Col1': [1, 2, 3, 4],
        'Col2': [4, 1, 2, 3],
        'Col3': [1, 1, 1, 1],
        'Col4': [2, 2, 3, 4]
    })

    incomplete_data = pd.DataFrame({
        'Col1': [1, 2, np.nan, 4],
        'Col2': [np.nan, 1, 2, 3],
        'Col3': [1, np.nan, 1, 1],
        'Col4': [2, 2, 3, np.nan]
    })
    return complete_data, incomplete_data

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

def load_dataset(datasetid, missing_rate=0.05):
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
        logging.warning(f"Target columns are missing or not valid for dataset {dataset_id}. Proceeding without dropping any columns.")
        df_dropped = df  # If no valid target columns, don't drop any columns

    # Use df_dropped as complete_data (without missing values)
    complete_data = df_dropped.copy()

    # Generate missing values for incomplete_data using the original copy of df_dropped
    incomplete_data = generate_missing_df(df_dropped, missing_rate)  # Generate missing values for incomplete_data

    # Ensure complete_data contains no missing values
    complete_values_count = complete_data.isna().sum().sum()
    logging.info(f"The complete DataFrame contains {complete_values_count} missing values after load_dataset()")

    # Check if incomplete_data contains missing values
    missing_values_count = incomplete_data.isna().sum().sum()
    logging.info(f"The incomplete DataFrame contains {missing_values_count} missing values after load_dataset()")

    # Return both the complete and incomplete datasets
    return complete_data, incomplete_data


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

    def train_with_logging(self, max_episodes=1000, max_total_steps=10000, log_interval=100, log_dir="./logs", file_path="./final_results.csv", experiment_table=None, progress_data=None, index=None):
        start_episode = 1
        total_steps = 0  # Track the total number of steps (iterations)

        try:
            while start_episode <= max_episodes and total_steps < max_total_steps:  # Stop after max_episodes or total steps reached
                logging.info(f"Starting Episode {start_episode}/{max_episodes}")

                # Reset environment at the start of each episode
                state = self.env.reset()
                done = False
                step = 0

                # Loop over steps within an episode until done or max_total_steps is reached
                while not done and total_steps < max_total_steps:
                    position = random.choice(self.env.missing_indices)
                    action = self.choose_action(position)
                    next_state, reward, done = self.env.step(action, position)
                    self.learn(position, action, reward, next_state)
                    state = next_state
                    step += 1
                    total_steps += 1  # Track total number of steps

                    # Decay epsilon
                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                # Log at the end of each episode
                if start_episode % log_interval == 0:
                    logging.info(f"Episode {start_episode} completed with {step} steps. Total steps so far: {total_steps}/{max_total_steps}")

                start_episode += 1

            # Calculate MAE and RMSE once the entire imputation is done
            mae, rmse = calculate_metrics(self.env, self)
            logging.info(f"Final MAE = {mae:.6f}, Final RMSE = {rmse:.6f}")

            # Update Streamlit table with MAE and RMSE at the end of all episodes
            if experiment_table is not None and progress_data is not None:
                progress_data.at[index, 'MAE'] = mae
                progress_data.at[index, 'RMSE'] = rmse
                experiment_table.dataframe(progress_data)

            # Save the final imputed data to the specified file path after all episodes
            self.env.state.to_csv(file_path, index=False)
            logging.info(f"Final imputed data saved to {file_path}")

        except KeyboardInterrupt:
            logging.info("Training interrupted.")


class ImputationEnvironment:
    def __init__(self, incomplete_data, complete_data):
        self.incomplete_data = incomplete_data.copy()
        self.complete_data = complete_data
        self.state = incomplete_data.copy()
        self.missing_indices = np.argwhere(pd.isna(self.incomplete_data.values))

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


# Define search spaces for alpha and gamma based on the paper
alpha_space = np.arange(0, 0.501, 0.001)  # Alpha: {0, 0.001, 0.002, ..., 0.5}
gamma_space = np.arange(0.9, 1.0, 0.01)   # Gamma: {0.9, 0.91, 0.92, ..., 0.99}

def grid_search_qlearning(dataset_id, missing_rate, max_episodes=1000, max_total_steps=10000):
    best_mae = float('inf')
    best_rmse = float('inf')
    best_alpha = None
    best_gamma = None
    best_results = {}

    # Loop through all combinations of alpha and gamma
    for alpha in alpha_space:
        for gamma in gamma_space:
            logging.info(f"Evaluating alpha={alpha}, gamma={gamma}")

            # Load dataset
            complete_data, incomplete_data = load_dataset(dataset_id, missing_rate)
            env = ImputationEnvironment(incomplete_data, complete_data)

            # Initialize RL agent with current alpha and gamma
            agent = RLImputer(env=env, alpha=alpha, gamma=gamma, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)

            # Run the Q-learning process
            agent.train_with_logging(max_episodes=max_episodes, max_total_steps=max_total_steps)

            # Calculate metrics (MAE, RMSE) after training
            mae, rmse = calculate_metrics(env, agent)

            logging.info(f"Results for alpha={alpha}, gamma={gamma}: MAE={mae}, RMSE={rmse}")

            # Store the best performing hyperparameters based on MAE or RMSE
            if mae < best_mae:
                best_mae = mae
                best_rmse = rmse
                best_alpha = alpha
                best_gamma = gamma
                best_results = {
                    "alpha": alpha,
                    "gamma": gamma,
                    "mae": mae,
                    "rmse": rmse
                }

    logging.info(f"Best Results: Alpha={best_alpha}, Gamma={best_gamma}, MAE={best_mae}, RMSE={best_rmse}")
    return best_results


# if __name__ == "__main__":
#     dataset_ids = [94, 59, 17, 332, 350, 189, 484, 149] # all datasets
#     missing_rates = [0.05, 0.10, 0.15, 0.20]  # our missing rates

#     for dataset_id in dataset_ids:
#         for missing_rate in missing_rates:
#             logging.info(f"Processing dataset ID: {dataset_id} with missing rate: {missing_rate}")

#             # Load the dataset
#             complete_data, incomplete_data = load_dataset(dataset_id, missing_rate)

#             # Initialize the environment and the Q-learning agent
#             env = ImputationEnvironment(incomplete_data, complete_data)
#             agent = RLImputer(env=env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)

#             # Define a specific checkpoint directory for each dataset and missing rate
#             checkpoint_dir = f"./checkpoints/dataset_{dataset_id}_missing_rate_{int(missing_rate * 100)}"

#             # Train the agent and create checkpoints
#             agent.train_with_logging(episodes=250, log_interval=50, checkpoint_dir=checkpoint_dir, patience=15)

#             logging.info(f"Completed training for dataset ID: {dataset_id} with missing rate: {missing_rate}")

#     logging.info("All datasets processed.")
