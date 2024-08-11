import sys
import argparse
import logging
import numpy as np
import pandas as pd
import random

# Import classes and utility functions from external files
from environment import ImputationEnvironment
from qlearning import QLearningAgent
from dqlearning import DQNAgent
from util import data_loader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# For detailed debugging information, use logging.DEBUG
logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Choose RL approach and dataset for imputation.")
    parser.add_argument('--method', type=str, choices=['qlearning', 'dqlearning'], default='qlearning',
                        help='Choose the RL method: qlearning or dqlearning')
    parser.add_argument('--episodes', type=int, default=100000,
                        help='Number of training steps for RL Agent')
    parser.add_argument('--id', type=int, help='ID of the UCI dataset to use for imputation'),
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Threshold for missing data imputation'),
    parser.add_argument('--incomplete_data', type=str, help='Path to incomplete data CSV file')
    parser.add_argument('--complete_data', type=str, help='Path to complete data CSV file')

    return parser.parse_args()


def rl_imputation(args):
    toy_data = False
    try:
        episodes = args.episodes
        method = args.method
        threshold = args.threshold
        if args.id is not None:
            datasetid = args.id
            logging.info(f"Loading dataset with ID {datasetid}")
            complete_data, incomplete_data = data_loader.load_dataset(datasetid, threshold)
        else:
            if args.incomplete_data == "data/toy_dataset_missing.csv" and args.complete_data == "data/toy_dataset.csv":
                toy_data = True
            incomplete_data = pd.read_csv(args.incomplete_data)
            complete_data = pd.read_csv(args.complete_data)
            logging.info(f"Data loaded from {args.incomplete_data} and {args.complete_data}.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    if toy_data:  # our toy dataset has "?" as missing values
        incomplete_data.replace("?", np.nan, inplace=True)
        complete_data.replace("?", np.nan, inplace=True)

    # Check for missing values after generation
    logging.info(
        f"Number of missing data in incomplete_data before preprocessing: {incomplete_data.isnull().sum().sum()}")

    # Preprocess the data
    if not toy_data:
        incomplete_data, complete_data = data_loader.preprocess_data(incomplete_data, complete_data)

    # Check for missing values after preprocessing
    logging.info(
        f"Number of missing data in incomplete_data after preprocessing: {incomplete_data.isnull().sum().sum()}")

    logging.info("Creating Imputation Environment.")
    env = ImputationEnvironment(incomplete_data, complete_data)

    if method == 'qlearning':
        logging.info("Using Q-Learning approach.")
        agent = QLearningAgent(env)
        agent.train(episodes=episodes)
        imputed_data = env.state
        print(imputed_data)
        imputed_data.to_csv("results/imputed_data.csv", index=False)
        if toy_data:
            # Ensure the DataFrames have the same shape
            if imputed_data.shape != complete_data.shape:
                raise ValueError("DataFrames do not have the same shape")

            # Ensure the DataFrames have the same columns
            if not (imputed_data.columns == complete_data.columns):
                raise ValueError("DataFrames do not have the same columns")

            # Compare the DataFrames element-wise, handling NaN values
            comparison = imputed_data.fillna(np.nan) == complete_data.fillna(np.nan)
            similarity_percentage = comparison.mean().mean() * 100

            print(f"Similarity Percentage: {similarity_percentage:.2f}%")
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
