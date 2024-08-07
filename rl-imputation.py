import numpy as np
import pandas as pd
import random
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

class ImputationEnvironment:
    def __init__(self, incomplete_data, complete_data):
        self.incomplete_data = incomplete_data
        self.complete_data = complete_data
        self.state = incomplete_data.copy()
        self.missing_indices = np.argwhere(pd.isna(incomplete_data.values))

    def reset(self):
        """Reset the environment state to the initial incomplete data."""
        self.state = self.incomplete_data.copy()
        return self.state

    def step(self, action, position):
        """Take an action to impute a missing value at the given position."""
        row, col = position
        self.state.iat[row, col] = action

        # Reward is the negative absolute error
        reward = -abs(self.complete_data.iat[row, col] - action)
        done = not pd.isna(self.state.values).any()  # Check if all values are imputed
        return self.state, reward, done

    def get_possible_actions(self, col):
        """Get unique possible values for a column."""
        col = int(col)  # Ensure col is an integer
        if 0 <= col < len(self.complete_data.columns):
            col_name = self.complete_data.columns[col]
        else:
            raise KeyError(f"Column index {col} out of range")

        return self.complete_data[col_name].dropna().unique()

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))

    def choose_action(self, state, position):
        """Choose an action using an epsilon-greedy policy."""
        state_key = (tuple(state.values.flatten()), tuple(position))

        if random.uniform(0, 1) < self.epsilon:
            # Exploration: choose a random action
            return random.choice(self.env.get_possible_actions(position[1]))
        else:
            # Exploitation: choose the best known action
            col = position[1]
            actions = self.env.get_possible_actions(col)
            q_values = {a: self.q_table[state_key][a] for a in actions}
            return max(q_values, key=q_values.get)

    def learn(self, state, action, reward, next_state, position):
        """Update the Q-table based on the action taken."""
        state_key = (tuple(state.values.flatten()), tuple(position))
        next_state_key = (tuple(next_state.values.flatten()), tuple(position))

        q_predict = self.q_table[state_key][action]
        q_target = reward + self.gamma * max(self.q_table[next_state_key].values(), default=0)
        self.q_table[state_key][action] += self.alpha * (q_target - q_predict)

    def train(self, episodes=1000):
        """Train the agent over a number of episodes."""
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                position = random.choice(self.env.missing_indices)
                action = self.choose_action(state, position)
                next_state, reward, done = self.env.step(action, position)
                self.learn(state, action, reward, next_state, position)
                state = next_state

# Load data and train the agent
incomplete_data_path = 'data/toy_dataset_missing.csv'
complete_data_path = 'data/toy_dataset.csv'

incomplete_data = pd.read_csv(incomplete_data_path)
complete_data = pd.read_csv(complete_data_path)

incomplete_data.replace("?", np.nan, inplace=True)
complete_data.replace("?", np.nan, inplace=True)  # Ensure data is clean

# Optional: Scale the data
scaler = MinMaxScaler()
incomplete_data = pd.DataFrame(scaler.fit_transform(incomplete_data), columns=incomplete_data.columns)
complete_data = pd.DataFrame(scaler.transform(complete_data), columns=complete_data.columns)

env = ImputationEnvironment(incomplete_data, complete_data)
agent = QLearningAgent(env)

agent.train(episodes=300000)

# Imputed data
imputed_data = env.state
print(imputed_data)
