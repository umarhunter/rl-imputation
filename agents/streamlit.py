import streamlit as st
import pandas as pd
import logging
from qlearning import load_dataset, RLImputer, ImputationEnvironment

# Initialize Streamlit UI
st.title('Q-learning Imputation Experiment Dashboard')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Placeholder for displaying progress table
experiment_table = st.empty()

# Track progress in a DataFrame
progress_data = pd.DataFrame(columns=['Dataset ID', 'Missingness', 'Status'])

# List of datasets and missingness rates
# dataset_ids = [94, 59, 17, 332, 350, 189, 484, 149]  # UCI dataset IDs
dataset_ids = [94, 59, 17, 350, 189, 149]  # UCI dataset IDs

missingness_rates = [0.05, 0.10, 0.15, 0.20]

# Track progress in a DataFrame
rows = []
for dataset_id in dataset_ids:
    for rate in missingness_rates:
        rows.append({
            'Dataset ID': dataset_id, 
            'Missingness': rate, 
            'Status': 'Not Started',
            'MAE': None,
            'RMSE': None
        })

# Use pd.DataFrame() to create the progress tracker
progress_data = pd.DataFrame(rows)

# Placeholder for displaying progress table
experiment_table = st.empty()


# Iterate over datasets and missing rates
for index, row in progress_data.iterrows():
    if row['Status'] == 'Not Started':
        dataset_id = row['Dataset ID']
        missing_rate = row['Missingness']

        # Load the dataset and initialize environment
        complete_data, incomplete_data = load_dataset(dataset_id, missing_rate)
        env = ImputationEnvironment(incomplete_data, complete_data)
        agent = RLImputer(env=env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)

        # Update the progress to "In Progress"
        progress_data.at[index, 'Status'] = 'In Progress'
        experiment_table.dataframe(progress_data)  # Update the displayed table
        
        # Run the experiment
        file_path = f"./results/dataset_{dataset_id}_missing_{int(missing_rate * 100)}.csv"
        agent.train_with_logging(max_episodes=1000, max_steps_per_episode=10000, file_path=file_path, experiment_table=experiment_table, progress_data=progress_data, index=index)
        
        # Mark as completed
        progress_data.at[index, 'Status'] = 'Completed'
        experiment_table.dataframe(progress_data)  # Update the displayed table again

        # Log completion
        logging.info(f"Completed experiment on dataset {dataset_id} with missing rate {missing_rate}")

# Final message
st.success("All experiments completed!")

# Option to save the results to CSV
if st.button('Save Results'):
    progress_data.to_csv("experiment_progress.csv", index=False)
    st.write("Progress saved to experiment_progress.csv")
