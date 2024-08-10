import sys
import argparse
import logging
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

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
                        help='Threshold for missing data imputation'),
    parser.add_argument('--incomplete_data', type=str, help='Path to incomplete data CSV file')
    parser.add_argument('--complete_data', type=str, help='Path to complete data CSV file')

    return parser.parse_args()


def preprocess_data(incomplete_data, complete_data):
    # Identify categorical and numerical columns
    categorical_columns = incomplete_data.select_dtypes(include=['object']).columns
    numerical_columns = incomplete_data.select_dtypes(exclude=['object']).columns

    # Encode categorical variables using Label Encoding
    ## NOTE: Not sure if I should use label encoding or one-hot encoding
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        incomplete_data[col] = le.fit_transform(incomplete_data[col].astype(str))
        complete_data[col] = le.transform(complete_data[col].astype(str))
        label_encoders[col] = le

    # Scale numerical variables
    scaler = MinMaxScaler()
    incomplete_data[numerical_columns] = scaler.fit_transform(incomplete_data[numerical_columns])
    complete_data[numerical_columns] = scaler.transform(complete_data[numerical_columns])

    return incomplete_data, complete_data


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

    # Preprocess the data
    incomplete_data, complete_data = preprocess_data(incomplete_data, complete_data)

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
