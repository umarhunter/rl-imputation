import logging
import gymnasium as gym
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


class ImputationEnv(gym.Env):
    def __init__(self, incomplete_data, complete_data):
        super(ImputationEnv, self).__init__()
        self.incomplete_data = incomplete_data.copy()
        self.complete_data = complete_data.copy()
        self.counter = 0
        # Find the missing indices
        self.missing_indices = np.argwhere(pd.isnull(self.incomplete_data).values)
        self.current_index = 0

        # Check if there are missing values
        if len(self.missing_indices) == 0:
            raise ValueError("No missing values found in the dataset for imputation.")

        # Define the range of possible actions
        self.min_value = self.complete_data.min().min()
        self.max_value = self.complete_data.max().max()
        self.num_actions = 100  # Increased from 10 to allow finer-grained actions


        # Define action and observation spaces
        self.observation_space = gym.spaces.Box(
            low=self.min_value, high=self.max_value, shape=(self.incomplete_data.shape[1],), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.num_actions)
        #self.action_space = gym.spaces.Box(low=self.min_value, high=self.max_value, shape=(1,), dtype=np.float32)
        # Create a scaler for normalization
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(self.complete_data.fillna(0))

        # Precision map containing the max precision for each column, preventing loss of precision during imputation
        self.precision_map = {}
        for col in self.complete_data.columns:
            max_precision = 0
            for value in self.complete_data[col].dropna():
                if '.' in str(value):
                    decimal_places = len(str(value).split('.')[1])
                    if decimal_places > max_precision:
                        max_precision = decimal_places
            self.precision_map[col] = max_precision

    def reset(self, seed=None, options=None):
        # Set the random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset the environment's internal state
        self.current_index = 0
        self.counter = 0  # Reset any counters or additional state variables

        # Optionally, you could shuffle missing indices if the order should vary between episodes
        np.random.shuffle(self.missing_indices)

        # Return the initial observation and an empty info dictionary
        initial_observation = self._get_observation()
        return initial_observation, {}

    def _get_observation(self):
        # Check if the current_index is within the valid range
        if self.current_index >= len(self.missing_indices):
            # Return a default observation or handle it gracefully
            # You might choose to return zeros or the last valid observation
            #print(f"Warning: current_index {self.current_index} is out of bounds, returning default observation.")
            default_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return default_obs

        # Proceed with normal observation retrieval if the index is valid
        row, col = self.missing_indices[self.current_index]

        obs = self.incomplete_data.iloc[[row]].fillna(0).infer_objects(copy=False)
        obs = self.scaler.transform(obs)[0]
        return obs.astype(np.float32)

    def step(self, action):
        # Check if the environment is done before processing the action
        if self.current_index >= len(self.missing_indices):
            terminated = True
            truncated = False  # Adjust according to your time limit logic
            self.counter += 1

            # Return the final valid observation, with done=True
            last_valid_obs = self._get_observation() if self.current_index > 0 else np.zeros(
                self.observation_space.shape)
            return last_valid_obs, 0.0, terminated, truncated, {}

        # Proceed normally if not done
        row, col = self.missing_indices[self.current_index]

        action_value = self.min_value + (action / (self.num_actions - 1)) * (self.max_value - self.min_value)

        # Round the predicted value based on the column's precision
        precision = self.precision_map[self.incomplete_data.columns[col]]
        predicted_value = round(action_value, precision)

        actual_value = self.complete_data.iloc[row, col]

        reward = -abs(predicted_value - actual_value)  # Penalize based on the error
        self.incomplete_data.iloc[row, col] = predicted_value

        print(f"Imputed value: {predicted_value} at row: {row}, column: {self.incomplete_data.columns[col]}")

        self.current_index += 1
        done = self.current_index >= len(self.missing_indices)

        # Return the observation, reward, done flag, and info
        return self._get_observation(), reward, done, False, {}

    def render(self, mode='human'):
        pass


class ImputationEnvironment:
    def __init__(self, incomplete_data, complete_data):
        self.incomplete_data = incomplete_data
        self.complete_data = complete_data
        self.state = incomplete_data.copy()
        self.missing_indices = np.argwhere(pd.isna(incomplete_data.values))
        if len(self.missing_indices) == 0:
            logging.warning("Environment initialized with no missing indices.")

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
