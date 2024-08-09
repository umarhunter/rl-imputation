import numpy as np
import pandas as pd
import random
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from environment import ImputationEnvironment


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
