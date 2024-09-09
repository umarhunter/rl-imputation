import logging
import gymnasium as gym
import numpy as np
import pandas as pd


class ImputationEnv(gym.Env):
    def __init__(self, incomplete_data, complete_data, scaler):
        super(ImputationEnv, self).__init__()
        self.incomplete_data = incomplete_data.copy()
        self.complete_data = complete_data.copy()
        self.scaler = scaler  # the scaler for inverse transformation
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

        # Increase the number of discrete actions for finer granularity
        self.action_space = gym.spaces.Discrete(self.num_actions)

    def reset(self, seed=None, options=None):
        # Set the random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset the environment's internal state
        self.current_index = 0

        # Optionally, you could shuffle missing indices if the order should vary between episodes
        np.random.shuffle(self.missing_indices)

        # Return the initial observation and an empty info dictionary
        initial_observation = self._get_observation()
        return initial_observation, {}

    def _get_observation(self):
        # Check if the current_index is within the valid range
        if self.current_index >= len(self.missing_indices):
            # return a default observation or handle it
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
            truncated = False  # additional flag to indicate if the episode was truncated
            self.counter += 1

            # Return the final valid observation, with done=True
            last_valid_obs = self._get_observation() if self.current_index > 0 else np.zeros(
                self.observation_space.shape)
            return last_valid_obs, 0.0, terminated, truncated, {}

        # Proceed normally if not done
        row, col = self.missing_indices[self.current_index]

        # Calculate the action value based on the action taken by the agent (in scaled space)
        action_value = self.min_value + (action / (self.num_actions - 1)) * (self.max_value - self.min_value)

        # Use the scaled space for both imputation and comparison
        imputed_value_scaled = action_value  # The action value is in the scaled space

        # Get the actual scaled value from complete_data (already scaled)
        actual_value_scaled = self.complete_data.iloc[row, col]

        # Compare the imputed value and actual value in the scaled space
        reward = -abs(imputed_value_scaled - actual_value_scaled) ** 2

        # Store the imputed value in the incomplete data (in the scaled space)
        self.incomplete_data.iloc[row, col] = imputed_value_scaled

        # print(f"Imputed value (scaled): {imputed_value_scaled} at row: {row}, column: {self.incomplete_data.columns[col]}")
        # print(f"Actual value (scaled): {actual_value_scaled} at row: {row}, column: {self.incomplete_data.columns[col]}")

        self.current_index += 1
        done = self.current_index >= len(self.missing_indices)

        # Return the observation, reward, done flag, and info
        return self._get_observation(), reward, done, False, {}

    def finalize_imputation(self):
        """Inverse transform the scaled data back to the original scale."""
        logging.info(f"First 5 rows of incomplete data after completing imputation:\n{self.incomplete_data.head(5)}")

        # Ensure the data structure is consistent with what the scaler was fitted on
        if self.incomplete_data.shape != self.complete_data.shape:
            raise ValueError("Shape of incomplete data does not match complete data for inverse transformation.")

        # Fill NaNs temporarily and apply inverse transform
        incomplete_data_filled = self.incomplete_data.fillna(0)

        # Apply inverse transform to the scaled data
        imputed_data_original_scale = pd.DataFrame(
            self.scaler.inverse_transform(incomplete_data_filled),
            columns=self.incomplete_data.columns
        )

        logging.info(f"First 5 rows of imputed data after inverse:\n{imputed_data_original_scale.head(5)}")
        return imputed_data_original_scale

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
