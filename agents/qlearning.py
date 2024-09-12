import logging
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from skrl.agents.torch.q_learning import Q_LEARNING, Q_LEARNING_DEFAULT_CONFIG
from util import data_loader, util

# Set the root logger to INFO level to suppress debug logs from other libraries
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QImputationEnvironment(gym.Env):
    def __init__(self, incomplete_data, complete_data, num_bins=5):
        super(QImputationEnvironment, self).__init__()

        self.complete_data = complete_data
        self.incomplete_data = incomplete_data
        self.num_bins = num_bins

        # Find missing value indices
        self.missing_indices = np.argwhere(pd.isnull(incomplete_data.values))
        self.current_index = 0

        # Two possible actions: increase or decrease the imputed value
        self.action_space = gym.spaces.Discrete(2)

        # Observation space represents the imputation error
        self.observation_space = gym.spaces.Discrete(self.num_bins)

        # Min and max values in the complete dataset
        self.min_value = complete_data.min().min()
        self.max_value = complete_data.max().max()

    def discretize_error(self, actual, imputed):
        """Convert the imputation error into a discrete observation."""
        error = abs(actual - imputed)
        error_bins = np.linspace(0, self.max_value - self.min_value, self.num_bins)
        return np.digitize(error, error_bins) - 1

    def reset(self):
        """Reset the environment and start with the first missing value."""
        self.current_index = 0
        if len(self.missing_indices) == 0:
            raise ValueError("No missing values found in the dataset.")
        row_idx, col_idx = self.missing_indices[self.current_index]
        self.current_value = self.incomplete_data.iat[row_idx, col_idx]
        return 0  # Start with zero error

    def step(self, action):
        """Apply the action (increase or decrease) to impute the missing value."""
        row_idx, col_idx = self.missing_indices[self.current_index]
        actual_value = self.complete_data.iat[row_idx, col_idx]

        # Modify the imputed value based on the action
        if action == 0:  # Decrease the imputed value
            self.current_value = max(self.min_value, self.current_value - (self.max_value - self.min_value) / self.num_bins)
        else:  # Increase the imputed value
            self.current_value = min(self.max_value, self.current_value + (self.max_value - self.min_value) / self.num_bins)

        # Calculate reward as negative absolute error
        reward = -abs(actual_value - self.current_value)

        # Move to the next missing value
        self.current_index += 1
        done = self.current_index >= len(self.missing_indices)

        # Get the next observation (discretized error)
        if not done:
            next_row_idx, next_col_idx = self.missing_indices[self.current_index]
            next_actual_value = self.complete_data.iat[next_row_idx, next_col_idx]
            next_obs = self.discretize_error(next_actual_value, self.current_value)
        else:
            next_obs = None

        return next_obs, reward, done, {}

    def finalize_imputation(self):
        """Return the imputed dataset."""
        return self.incomplete_data


class TabularPolicy:
    def __init__(self, q_table, action_space, epsilon=0.1):
        self.q_table = q_table
        self.action_space = action_space
        self.epsilon = epsilon

    def act(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space.n)  # Explore: random action
        else:
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.action_space.n)
            return np.argmax(self.q_table[state])  # Exploit: best action


# Load datasets
complete_data, incomplete_data = data_loader.load_dataset(17, 0.1)

# Initialize the environment
env = QImputationEnvironment(incomplete_data, complete_data)

# Initialize Q-table
q_table = {}

# Instantiate the policy
policy = TabularPolicy(q_table, env.action_space)

# Q-learning parameters
learning_rate = 0.5
discount_factor = 0.99
num_episodes = 1000
epsilon = 0.1

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Select action
        action = policy.act(state)

        # Take the action and get the next state, reward, and done flag
        next_state, reward, done, _ = env.step(action)

        # Update Q-value
        if state not in q_table:
            q_table[state] = np.zeros(env.action_space.n)
        if next_state not in q_table:
            q_table[next_state] = np.zeros(env.action_space.n)

        q_value = q_table[state][action]
        max_next_q_value = np.max(q_table[next_state])
        q_table[state][action] = q_value + learning_rate * (reward + discount_factor * max_next_q_value - q_value)

        total_reward += reward
        state = next_state

    logging.info(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Imputed dataset
imputed_data = env.finalize_imputation()
print("Imputed Data:")
print(imputed_data)
