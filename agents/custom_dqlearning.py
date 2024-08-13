import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class DQNAgent:
    def __init__(self, env, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 learning_rate=0.001, batch_size=32, memory_size=2000):
        self.env = env
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
        possible_actions = self.env.get_possible_actions(position_col_index)
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
                possible_next_actions = self.env.get_possible_actions(position_col_index)
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
