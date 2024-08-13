import logging
import random
from collections import defaultdict

# Set the root logger to INFO level to suppress debug logs from other libraries
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        logging.info("Initializing Q-Learning Agent")
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
            action = random.choice(self.env.get_possible_actions(position[1]))
            #logging.info(f"Chose random action {action} for position {position}")
            return action
        else:
            # Exploitation: choose the best known action
            col = position[1]
            actions = self.env.get_possible_actions(col)
            q_values = {a: self.q_table[state_key][a] for a in actions}
            best_action = max(q_values, key=q_values.get)
            #logging.info(f"Chose best action {best_action} for position {position} with Q-values {q_values}")
            return best_action

    def learn(self, state, action, reward, next_state, position):
        """Update the Q-table based on the action taken."""
        state_key = (tuple(state.values.flatten()), tuple(position))
        next_state_key = (tuple(next_state.values.flatten()), tuple(position))

        q_predict = self.q_table[state_key][action]
        q_target = reward + self.gamma * max(self.q_table[next_state_key].values(), default=0)
        self.q_table[state_key][action] += self.alpha * (q_target - q_predict)

        #logging.info(f"Updated Q-value for state {state_key} and action {action}: {self.q_table[state_key][action]}")

    def train(self, episodes, log_interval=5000):
        """Train the agent over a number of episodes."""
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            step = 0
            while not done:
                position = random.choice(self.env.missing_indices)
                action = self.choose_action(state, position)
                next_state, reward, done = self.env.step(action, position)
                self.learn(state, action, reward, next_state, position)
                state = next_state
                step += 1

            # Log progress every 'log_interval' episodes
            if (episode + 1) % log_interval == 0:
                logging.info(f"Episode {episode + 1}/{episodes} completed with {step} steps.")
