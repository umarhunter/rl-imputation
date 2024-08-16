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
        self.num_actions = 10  # Number of discrete actions

        # Define action and observation spaces
        self.observation_space = gym.spaces.Box(
            low=self.min_value, high=self.max_value, shape=(self.incomplete_data.shape[1],), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.num_actions)

        # Create a scaler for normalization
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(self.complete_data.fillna(0))

        # # Pre-calculate precision for each column
        # self.precision_map = {
        #     col: self.complete_data[col].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0).max()
        #     for col in self.complete_data.columns
        # }
    def reset(self, seed=None, options=None):
        # Set the random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset the environment's internal state
        self.current_index = 0
        initial_observation = self._get_observation()

        # Return the initial observation and an empty info dictionary
        return initial_observation, {}

    def _get_observation(self):
        # if self.current_index >= len(self.missing_indices):
        #     print(
        #         f"current_index: {self.current_index}, len(missing_indices): {len(self.missing_indices)}, complete_data: {self.complete_data.shape}")
        #     raise IndexError(
        #         f"current_index {self.current_index} is out of bounds for the missing indices with length {len(self.missing_indices)}")

        row, col = self.missing_indices[self.current_index]

        # will result in
        # obs = self.incomplete_data.iloc[row].fillna(0).values
        # obs = self.scaler.transform([obs])[0]
        obs = self.incomplete_data.iloc[[row]].fillna(0)
        obs = self.scaler.transform(obs)[0]
        return obs.astype(np.float32)

    def step(self, action):
        # Ensure current index is within bounds
        if self.current_index >= len(self.missing_indices):
            raise IndexError(f"Current index {self.current_index} is out of bounds in missing indices.")

        # Get the row and column index of the missing value
        row, col = self.missing_indices[self.current_index]

        # Map the action to a real value
        action_value = self.min_value + (action / (self.num_actions - 1)) * (self.max_value - self.min_value)
        predicted_value = action_value

        # Get the actual value from the complete data
        actual_value = self.complete_data.iloc[row, col]

        # # Retrieve the pre-calculated precision for the column
        # column_name = self.incomplete_data.columns[col]
        # precision = self.precision_map[column_name]
        #
        # # Round the imputed value to match the original data's precision
        # if precision > 0:
        #     predicted_value = round(predicted_value, precision)
        # else:
        #     predicted_value = round(predicted_value)

        # predicted_value = round(predicted_value, 2)

        # Calculate reward based on accuracy of prediction
        reward = -abs(predicted_value - actual_value)  # Penalize based on the error

        # Apply the imputation
        self.incomplete_data.iloc[row, col] = predicted_value

        # Print the imputed value with row and column information
        print(f"Imputed value: {predicted_value} at row: {row}, column: {self.incomplete_data.columns[col]}")

        # Move to the next missing value
        self.current_index += 1
        done = self.current_index >= len(self.missing_indices)

        # Set terminated and truncated
        terminated = done
        truncated = False  # Assuming no time limit is set, you can set it according to your use case

        if done:
            self.current_index -= 1  # Adjust index to prevent out-of-bounds error
            self.counter += 1
            return self._get_observation(), 0.0, terminated, truncated, {}

        return self._get_observation(), reward, terminated, truncated, {}

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
