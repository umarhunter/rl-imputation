import numpy as np
import pandas as pd


class ImputationEnvironment:
    def __init__(self, incomplete_data, complete_data):
        self.incomplete_data = incomplete_data
        self.complete_data = complete_data
        self.state = incomplete_data.copy()
        self.missing_indices = np.argwhere(pd.isna(incomplete_data.values))

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
