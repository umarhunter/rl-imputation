import sys
import argparse
import logging
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler

# Import classes and utility functions from external files
from qlearning import ImputationEnvironment, QLearningAgent
from dqlearning import DQNAgent
from util import util, data_loader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Choose RL approach and dataset for imputation.")
    parser.add_argument('--method', type=str, choices=['qlearning', 'dqlearning'], default='qlearning',
                        help='Choose the RL method: qlearning or dqlearning')
    parser.add_argument('--episodes', type=int, default=100000,
                        help='Number of training steps for RL Agent')
    parser.add_argument('--id', type=int, help='ID of the UCI dataset to use for imputation'),
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='ID of the UCI dataset to use for imputation'),
    parser.add_argument('--incomplete_data', type=str, help='Path to incomplete data CSV file')
    parser.add_argument('--complete_data', type=str, help='Path to complete data CSV file')

    return parser.parse_args()


def rl_imputation(args):
    try:
        episodes = args.episodes
        method = args.method
        threshold = args.threshold
        if args.id is not None:
            datasetid = args.id
            logging.info(f"Loading dataset with ID {datasetid}")
            incomplete_data, complete_data = data_loader.load_dataset(datasetid, threshold)
        else:
            incomplete_data = pd.read_csv(args.incomplete_data)
            complete_data = pd.read_csv(args.complete_data)
            logging.info(f"Data loaded from {args.incomplete_data} and {args.complete_data}.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    incomplete_data.replace("?", np.nan, inplace=True)
    complete_data.replace("?", np.nan, inplace=True)

    # Normalize the data
    scaler = MinMaxScaler()
    incomplete_data = pd.DataFrame(scaler.fit_transform(incomplete_data), columns=incomplete_data.columns)
    complete_data = pd.DataFrame(scaler.transform(complete_data), columns=complete_data.columns)

    env = ImputationEnvironment(incomplete_data, complete_data)

    if method == 'qlearning':
        logging.info("Using Q-Learning approach.")
        agent = QLearningAgent(env)
        agent.train(episodes=episodes)
        imputed_data = env.state
        print(imputed_data)
    elif method == 'dqlearning':
        logging.info("Using Deep Q-Learning approach.")
        state_size = incomplete_data.size
        action_size = max(len(env.get_possible_actions(col)) for col in range(incomplete_data.shape[1]))
        agent = DQNAgent(env, state_size=state_size, action_size=action_size)  # Pass env to DQNAgent

        for e in range(episodes):
            state = env.reset()
            state = state.values.flatten()
            done = False
            while not done:
                position = random.choice(env.missing_indices)
                position_col_index = position[1]
                action_index, action_value = agent.act(state, position_col_index)
                next_state, reward, done = env.step(action_value, position)
                next_state = next_state.values.flatten()
                agent.remember(state, action_index, reward, next_state, done, position_col_index)
                state = next_state
            agent.replay()

        agent.save("results/dqn_model.pth")
        imputed_data = env.state
        print(imputed_data)


def main():
    args = parse_args()

    # Check for conflicting arguments
    if args.id is not None and (args.incomplete_data or args.complete_data):
        print("Error: --id cannot be used with --incomplete_data or --complete_data")
        sys.exit(1)

    rl_imputation(args)


if __name__ == '__main__':
    main()
