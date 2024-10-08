import logging
import random
import pickle
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import torch

from ucimlrepo import fetch_ucirepo, list_available_datasets
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure logging to write to both console and file
log_filename = "training_log.log"
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

    # Calculate MAE and RMSE between imputed data and complete data
    mae = mean_absolute_error(env.complete_data.values.flatten(), imputed_data.values.flatten())
    mse = mean_squared_error(env.complete_data.values.flatten(), imputed_data.values.flatten())
    rmse = np.sqrt(mse)

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
    dataset_ids = [94, 59, 17, 332, 350, 189, 484, 149]
    dataset = fetch_ucirepo(id=94)
    df = dataset.data.original

    # Drop the target columns before generating missing values
    target_columns = dataset.metadata.target_col
    logging.info(f"Target columns: {target_columns}")

    # Drop the target columns before generating missing values
    df_dropped = df.drop(columns=target_columns)
    logging.info(f"Dropped target columns: {target_columns}")

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



class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        logging.info("Initializing Q-Learning Agent")
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
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

    def train_with_logging(self, episodes=10000, log_interval=100, log_dir="./logs",
                           checkpoint_dir="./checkpoints", resume=False, patience=50, delta=0.001):
        writer = SummaryWriter(log_dir)
        start_episode = 1
        best_rmse = float('inf')
        best_mae = float('inf')
        patience_counter = 0

        if resume:
            checkpoint = load_checkpoint(checkpoint_dir)
            if checkpoint:
                self.q_table = checkpoint['q_table']
                self.epsilon = checkpoint['epsilon']
                start_episode = checkpoint['episode'] + 1

        try:
            for episode in range(start_episode, episodes + 1):
                state = self.env.reset()
                done = False
                step = 0
                while not done:
                    position = random.choice(self.env.missing_indices)
                    action = self.choose_action(position)
                    next_state, reward, done = self.env.step(action, position)
                    self.learn(position, action, reward, next_state)
                    state = next_state
                    step += 1

                    # Decay epsilon
                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                if (episode) % log_interval == 0:
                    logging.info(f"Episode {episode}/{episodes} completed with {step} steps.")

                    writer.add_scalar("Episode/Steps", step, episode)
                    writer.add_scalar("Episode/Epsilon", self.epsilon, episode)

                    mae, rmse = calculate_metrics(self.env, self)
                    logging.info(f"Episode {episode}: MAE = {mae:.6f}, RMSE = {rmse:.6f}, EPSILON: {self.epsilon:.6f}")
                    writer.add_scalar("Metrics/MAE", mae, episode)
                    writer.add_scalar("Metrics/RMSE", rmse, episode)
                    save_checkpoint(self, episode, self.env.state, checkpoint_dir)
                    if rmse < best_rmse - delta or mae < best_mae - delta:
                        logging.info(
                            f"Metrics improved for episode: {episode}. RMSE: {best_rmse:.6f} -> {rmse:.6f}, MAE: {best_mae:.6f} -> {mae:.6f}. Resetting patience counter.")
                        best_rmse = rmse
                        best_mae = mae
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        logging.info(f"Metrics did not improve. Patience counter: {patience_counter}/{patience}")

                    if patience_counter >= patience:
                        logging.info(f"Stopping early at episode {episode} due to lack of improvement.")
                        break



        except KeyboardInterrupt:
            logging.info("Training interrupted. Saving final checkpoint.")
            save_checkpoint(self, episode, checkpoint_dir)
        finally:
            writer.close()

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

if __name__ == "__main__":
    #dataset_ids = [94, 59, 17, 332, 350, 189, 484, 149]
    dataset_ids = [59, 332, 350, 189, 484, 149]
    missing_rate = 0.05  # Set missing rate for all datasets

    for dataset_id in dataset_ids:
        logging.info(f"Processing dataset ID: {dataset_id}")

        # Load the dataset
        complete_data, incomplete_data = load_dataset(dataset_id, missing_rate)

        # Initialize the environment and the Q-learning agent
        env = ImputationEnvironment(incomplete_data, complete_data)
        agent = QLearningAgent(env=env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)

        # Define a specific checkpoint directory for each dataset
        checkpoint_dir = f"./checkpoints/dataset_{dataset_id}"

        # Train the agent and create checkpoints
        agent.train_with_logging(episodes=250, log_interval=50, checkpoint_dir=checkpoint_dir, patience=15)

        logging.info(f"Completed training for dataset ID: {dataset_id}")

    logging.info("All datasets processed.")
