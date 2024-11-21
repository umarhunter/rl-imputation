import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Assume load_dataset and split_dataset are predefined functions
def load_dataset(id):
    # For the sake of example, we'll generate synthetic data.
    # In practice, this function should load and return the dataset corresponding to 'id'.
    data = pd.read_csv('data/toy_dataset.csv')
    return data

def introduce_missingness(X, missing_fraction=0.1):
    """
    Introduce missingness into the DataFrame X by setting a fraction of its elements to NaN.
    """
    X_missing = X.copy()
    n_samples, n_features = X_missing.shape
    n_missing = int(np.floor(missing_fraction * n_samples * n_features))
    
    # Randomly choose flat indices for missing values
    missing_indices = np.random.choice(n_samples * n_features, n_missing, replace=False)
    
    # Convert flat indices to 2D indices (row, column)
    rows = missing_indices // n_features
    cols = missing_indices % n_features
    
    # Assign NaN to the selected indices
    X_missing.values[rows, cols] = np.nan
    
    return X_missing

class QLearningImputer:
    def __init__(self, n_states=10, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99,
                 min_epsilon=0.01, r=0.01, num_iterations=10000):
        self.n_states = n_states
        self.states = np.arange(n_states)
        self.actions = [0, 1]  # 0: decrease, 1: increase
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = None           # Q-table to be initialized later
        self.r = r              # Update rate for imputed value adjustment
        self.num_iterations = num_iterations

    def get_state(self, error):
        # Map the error to a discrete state index
        error = np.clip(error, 0, 1 - 1e-6)
        state = int(error * self.n_states)
        if state >= self.n_states:
            state = self.n_states - 1
        return state

    def select_action(self, state):
        # Epsilon-greedy strategy for action selection
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = np.argmax(self.Q[state])
        return action

    def update_Q(self, state, action, reward, next_state):
        # Update Q-table using the Q-learning update rule
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_delta = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_delta

    def train(self, X_train):
        n_samples, n_features = X_train.shape
        self.Q = np.zeros((self.n_states, len(self.actions)))
        for iteration in range(self.num_iterations):
            # For each sample in X_train
            for i in range(n_samples):
                x = X_train[i].copy()
                missing_indices = np.where(np.isnan(x))[0]
                for idx in missing_indices:
                    true_value = self.X_complete[i, idx]
                    est_value = self.column_means[idx]
                    max_value = self.feature_max[idx]
                    min_value = self.feature_min[idx]
                    error = abs(est_value - true_value) / (max_value - min_value + 1e-6)
                    state = self.get_state(error)
                    done = False
                    while not done:
                        action = self.select_action(state)
                        # Apply action to update estimate
                        if action == 0:
                            est_value *= (1 - self.r)
                        else:
                            est_value *= (1 + self.r)
                        # Calculate new error and next state
                        error = abs(est_value - true_value) / (max_value - min_value + 1e-6)
                        next_state = self.get_state(error)
                        # Compute reward
                        if next_state < state:
                            reward = 1
                        elif next_state == state:
                            reward = 0
                        else:
                            reward = -1
                        if next_state == 0:
                            reward = 10  # Reached goal state
                            done = True
                        # Update Q-table
                        self.update_Q(state, action, reward, next_state)
                        state = next_state
            # Decay epsilon after each iteration
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
            
            # Impute the training data using current Q-table
            X_imputed_train = self.impute(self.X_train_missing)
            # Evaluate imputation performance on training data
            mae = mean_absolute_error(self.X_complete[self.missing_mask],
                                      X_imputed_train[self.missing_mask])
            rmse = np.sqrt(mean_squared_error(self.X_complete[self.missing_mask],
                                              X_imputed_train[self.missing_mask]))
            logging.info(f'Iteration {iteration + 1}/{self.num_iterations}, MAE: {mae:.6f}, RMSE: {rmse:.6f} Epsilon: {self.epsilon:.6f}')

    def impute(self, X_missing):
        X_imputed = X_missing.copy()
        n_samples, n_features = X_imputed.shape
        for i in range(n_samples):
            x = X_imputed[i]
            missing_indices = np.where(np.isnan(x))[0]
            for idx in missing_indices:
                est_value = x[idx]
                if np.isnan(est_value):
                    # Start with the column mean
                    est_value = self.column_means[idx]
                    max_value = self.feature_max[idx]
                    min_value = self.feature_min[idx]
                    true_value = self.X_complete[i, idx]
                    error = abs(est_value - true_value) / (max_value - min_value + 1e-6)
                    state = self.get_state(error)
                    done = False
                    steps = 0
                    while not done:
                        # Select the best action based on the Q-table
                        action = np.argmax(self.Q[state])
                        # Apply action to update estimate
                        if action == 0:
                            est_value *= (1 - self.r)
                        else:
                            est_value *= (1 + self.r)
                        # Calculate new error and next state
                        error = abs(est_value - true_value) / (max_value - min_value + 1e-6)
                        next_state = self.get_state(error)
                        # Check if goal state is reached
                        if next_state == 0:
                            done = True
                        else:
                            state = next_state
                            steps += 1
                    x[idx] = est_value
        return X_imputed

    def fit(self, X_train_complete, X_train_missing):
        self.X_complete = X_train_complete
        self.X_train_missing = X_train_missing
        self.column_means = np.nanmean(X_train_missing, axis=0)
        self.feature_max = np.nanmax(X_train_complete, axis=0)
        self.feature_min = np.nanmin(X_train_complete, axis=0)
        self.missing_mask = np.isnan(X_train_missing)
        self.train(X_train_missing)

# Main script
if __name__ == '__main__':
    # Load dataset
    dataset_id = 1  # Replace with appropriate dataset ID
    data = load_dataset(dataset_id)
    complete_data = data.copy()

    # Introduce missing values into the complete data
    missing_fraction = 0.05
    data_with_missing = introduce_missingness(data, missing_fraction)

    # Split dataset into train and test sets
    train_data_missing, test_data_missing, train_data_complete, test_data_complete = train_test_split(
        data_with_missing, complete_data, test_size=0.3, random_state=42)

    # Convert to numpy arrays
    train_data_missing = train_data_missing.values
    test_data_missing = test_data_missing.values
    train_data_complete = train_data_complete.values
    test_data_complete = test_data_complete.values

    # Create QLearningImputer instance
    imputer = QLearningImputer(alpha=0.1, gamma=0.9, epsilon=1.0,
                               epsilon_decay=0.99, r=0.01, num_iterations=10000)

    # Fit the imputer on training data
    imputer.fit(train_data_complete, train_data_missing)

    # Impute missing values in test data
    test_data_imputed = imputer.impute(test_data_missing)

    # Evaluate imputation performance on test data
    missing_mask_test = np.isnan(test_data_missing)
    mae_test = mean_absolute_error(test_data_complete[missing_mask_test],
                                   test_data_imputed[missing_mask_test])
    rmse_test = np.sqrt(mean_squared_error(test_data_complete[missing_mask_test],
                                           test_data_imputed[missing_mask_test]))
    logging.info(f'Test MAE: {mae_test:.6f}, Test RMSE: {rmse_test:.6f}')
