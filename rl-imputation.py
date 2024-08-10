import argparse
import logging
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler

# Import classes from external files
from qlearning import ImputationEnvironment, QLearningAgent
from dqlearning import DQNAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def rl_imputation(args):
    # Load data
    try:
        incomplete_data = pd.read_csv(args.incomplete_data)
        complete_data = pd.read_csv(args.complete_data)
        numsteps = int(args.num_steps)
        logging.info(f"Data loaded from {args.incomplete_data} and {args.complete_data}.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    incomplete_data.replace("?", np.nan, inplace=True)
    complete_data.replace("?", np.nan, inplace=True)

    scaler = MinMaxScaler()
    incomplete_data = pd.DataFrame(scaler.fit_transform(incomplete_data), columns=incomplete_data.columns)
    complete_data = pd.DataFrame(scaler.transform(complete_data), columns=complete_data.columns)

    env = ImputationEnvironment(incomplete_data, complete_data)

    if args.method == 'qlearning':
        logging.info("Using Q-Learning approach.")
        agent = QLearningAgent(env)
        agent.train(episodes=numsteps)
        imputed_data = env.state
        print(imputed_data)
    elif args.method == 'dqlearning':
        logging.info("Using Deep Q-Learning approach.")
        state_size = incomplete_data.size
        action_size = max(len(env.get_possible_actions(col)) for col in range(incomplete_data.shape[1]))
        agent = DQNAgent(env, state_size=state_size, action_size=action_size)  # Pass env to DQNAgent

        EPISODES = numsteps
        for e in range(EPISODES):
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

        agent.save("dqn_model.pth")
        imputed_data = env.state
        print(imputed_data)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Choose RL approach and dataset for imputation.")
    parser.add_argument('--method', type=str, choices=['qlearning', 'dqlearning'], default='qlearning',
                        help='Choose the RL method: qlearning or dqlearning')
    parser.add_argument('--num_steps', type=int, default=500000,
                        help='Number of training steps for RL Agent')
    parser.add_argument('--incomplete_data', type=str, default='data/toy_dataset_missing.csv',
                        help='Path to incomplete data CSV file')
    parser.add_argument('--complete_data', type=str, default='data/toy_dataset.csv',
                        help='Path to complete data CSV file')
    args = parser.parse_args()

    rl_imputation(args)


if __name__ == '__main__':
    main()