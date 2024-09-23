import logging
import random
from collections import defaultdict
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def load_dataset(datasetid, missing_rate=0.10):
    # dataset = get_data(datasetid)
    # df = dataset.data.original
    df = pd.read_csv("breast_cancer_wisconsin.csv")
    # Hardcoded target columns for Breast Cancer Wisconsin dataset (drop first)
    if datasetid == 17:
        df = pd.read_csv("breast_cancer_wisconsin.csv")
        target_column = ['ID', 'Diagnosis']

        # Drop the target columns before generating missing values
        df_dropped = df.drop(columns=target_column)
        logging.info(f"Dropped target columns: {target_column}")

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

    # For other datasets, handle differently if needed (can keep this flexible for other cases)
    missing_df = generate_missing_df(df, missing_rate)
    missing_values_count = missing_df.isna().sum().sum()
    if missing_values_count > 0:
        logging.info(f"The DataFrame contains {missing_values_count} missing values after load_dataset()")

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01):
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

    def train(self, episodes, log_interval=100):
        logging.info(f"Training the Q-Learning Agent for {episodes} episodes.")
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            step = 0
            while not done:
                position = random.choice(self.env.missing_indices)
                action = self.choose_action(position)
                next_state, reward, done = self.env.step(action, position)
                next_position = position  # Simplified state transition
                self.learn(position, action, reward, next_position)
                step += 1

            # Decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if (episode + 1) % log_interval == 0:
                logging.info(f"Episode {episode + 1}/{episodes} completed with {step} steps.")
                logging.info(f"Epsilon after episode {episode + 1}: {self.epsilon}")

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

complete_data, incomplete_data = load_dataset(17, 0.10)
# Initialize the environment and the Q-learning agent
env = ImputationEnvironment(incomplete_data, complete_data)
agent = QLearningAgent(env=env, alpha=0.1, gamma=0.9, epsilon=0.1)

# Train the agent
agent.train(episodes=20000, log_interval=100)

# Signal completion of training
logging.info("Training completed.")
logging.info(f"Complete Data: \n{complete_data}")

logging.info("Training completed.")

# Output the imputed data
imputed_data = env.state
# logging.info(f"Imputed Data: \n{imputed_data}")
imputed_data.to_csv("imputed_data.csv", index=False)
