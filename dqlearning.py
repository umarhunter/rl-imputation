import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from sklearn.preprocessing import MinMaxScaler


class ImputationEnvironment:
    def __init__(self, incomplete_data, complete_data):
        self.incomplete_data = incomplete_data
        self.complete_data = complete_data
        self.state = incomplete_data.copy()
        self.missing_indices = np.argwhere(pd.isna(incomplete_data.values))

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
        col = int(col)
        if 0 <= col < len(self.complete_data.columns):
            col_name = self.complete_data.columns[col]
        else:
            raise KeyError(f"Column index {col} out of range")

        return self.complete_data[col_name].dropna().unique()


class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.001, batch_size=32, memory_size=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self):
        hidden_size = 24
        model = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size)  # Output one Q-value per action
        )
        return model

    def remember(self, state, action_index, reward, next_state, done, position_col_index):
        self.memory.append((state, action_index, reward, next_state, done, position_col_index))

    def act(self, state, position_col_index):
        possible_actions = env.get_possible_actions(position_col_index)
        if np.random.rand() <= self.epsilon:
            action_index = random.randrange(len(possible_actions))
        else:
            state_tensor = torch.FloatTensor(state)
            q_values = self.model(state_tensor)
            action_index = torch.argmax(q_values).item()
        return action_index, possible_actions[action_index]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for state, action_index, reward, next_state, done, position_col_index in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state)
                possible_next_actions = env.get_possible_actions(position_col_index)
                next_q_values = self.model(next_state_tensor)
                target = reward + self.gamma * torch.max(next_q_values).item()

            target_f = self.model(torch.FloatTensor(state))
            target_value = target_f.clone().detach()
            target_value[action_index] = target
            loss = criterion(target_f, target_value)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


# Load data
incomplete_data = pd.read_csv('data/toy_dataset_missing.csv')
complete_data = pd.read_csv('data/toy_dataset.csv')

incomplete_data.replace("?", np.nan, inplace=True)
complete_data.replace("?", np.nan, inplace=True)

scaler = MinMaxScaler()
incomplete_data = pd.DataFrame(scaler.fit_transform(incomplete_data), columns=incomplete_data.columns)
complete_data = pd.DataFrame(scaler.transform(complete_data), columns=complete_data.columns)

# Setup environment and agent
env = ImputationEnvironment(incomplete_data, complete_data)
state_size = incomplete_data.size  # Use total size if flattening entire data

# Assuming the maximum number of possible actions based on the column with the most unique values
action_size = max(len(env.get_possible_actions(col)) for col in range(incomplete_data.shape[1]))

agent = DQNAgent(state_size=state_size, action_size=action_size)

# Train the agent
EPISODES = 100000
for e in range(EPISODES):
    state = env.reset()
    state = state.values.flatten()
    done = False
    while not done:
        position = random.choice(env.missing_indices)
        position_col_index = position[1]  # Column index
        action_index, action_value = agent.act(state, position_col_index)
        next_state, reward, done = env.step(action_value, position)
        next_state = next_state.values.flatten()
        agent.remember(state, action_index, reward, next_state, done, position_col_index)
        state = next_state
    agent.replay()

# Save the trained model
agent.save("dqn_model.pth")

# Print the imputed data
dqlearning_imputed_data = env.state