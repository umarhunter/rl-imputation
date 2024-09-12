import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from ucimlrepo import fetch_ucirepo, list_available_datasets
from util.util import generate_missing_df


def print_prompt_box(prompt, content):
    # Define the dimensions of the box
    box_width = len(prompt) * 2  # 2 spaces padding on each side
    box_height = 3  # Top border, prompt line, bottom border

    # Print the top border
    print('*' * box_width)

    # Print the middle part with the prompt
    print('All datasets available in script')

    # Print the top border
    print('*' * box_width)

    print(content)

    # Print the bottom border
    print('*' * box_width)


def get_id():
    print_prompt_box("Enter the dataset ID",
                     "1. Spambase (94)\n"
                     "2. Letter Recognition (59)\n"
                     "3. Breast Cancer Wisconsin (Diag) (17)\n"
                     "4. Online News Popularity (332)\n"
                     "5. Default of Credit Card Clients (350)\n"
                     "6. Parkinsons Telemonitoring (189)\n"
                     "7. Travel Reviews (484)\n"
                     "8. Statlog (Vehicle Silhouettes) (149)")
    dataset_id = int(input("Enter the dataset ID: "))
    assert dataset_id in [94, 59, 17, 332, 350, 189, 484, 149], "Invalid dataset ID"
    return dataset_id


def get_data(dataset_id, X=False, y=False, variable_info=False, metadata=False):
    dataset = fetch_ucirepo(id=dataset_id)
    return dataset


def get_all_datasets():
    datasets = {"Spambase": 94, "Letter Recognition": 59, "Breast Cancer Wisconsin": 17, "Online News Popularity": 332,
                "Default Credit Card Clients": 350, "Parkinsons Telemonitoring": 189, "Travel Reviews": 484,
                "Statlog": 149}

    for dataset, num in datasets.items():
        df = get_data(num)
        df = df.data.original
        file_name = dataset.lower().replace(" ", "_") + ".csv"
        df.to_csv(file_name, index=False)


def load_dataset(datasetid, missing_rate=0.10):
    # dataset = get_data(datasetid)
    # df = dataset.data.original
    df = pd.read_csv("breast_cancer_wisconsin.csv")
    # Hardcoded target columns for Breast Cancer Wisconsin dataset (drop first)
    if datasetid == 17:
        df = pd.read_csv("breast_cancer_wisconsin.csv")
        target_column = ['ID', 'Diagnosis']

        # Drop the target columns before generating missing values
        df_dropped = df.drop(columns=target_column)
        logging.info(f"Dropped target columns: {target_column}")

        # Use df_dropped as complete_data (without missing values)
        complete_data = df_dropped.copy()

        # Generate missing values for incomplete_data using the original copy of df_dropped
        incomplete_data = generate_missing_df(df_dropped, missing_rate)  # Generate missing values for incomplete_data

        # Ensure complete_data contains no missing values
        complete_values_count = complete_data.isna().sum().sum()
        logging.info(f"The complete DataFrame contains {complete_values_count} missing values after load_dataset()")

        # Check if incomplete_data contains missing values
        missing_values_count = incomplete_data.isna().sum().sum()
        logging.info(f"The incomplete DataFrame contains {missing_values_count} missing values after load_dataset()")

        # Return both the complete and incomplete datasets
        return complete_data, incomplete_data

    # For other datasets, handle differently if needed (can keep this flexible for other cases)
    missing_df = generate_missing_df(df, missing_rate)
    missing_values_count = missing_df.isna().sum().sum()
    if missing_values_count > 0:
        logging.info(f"The DataFrame contains {missing_values_count} missing values after load_dataset()")


def preprocess_data(complete_data, incomplete_data):
    complete_data_original = complete_data.copy()

    # Check for missing values in complete_data
    if complete_data.isnull().sum().sum() > 0:
        raise ValueError(f"complete_data contains missing values: {complete_data.isnull().sum().sum()}")

    # # Log column names to ensure they match
    # logging.info(f"Columns in complete_data: {list(complete_data.columns)}")
    # logging.info(f"Columns in incomplete_data: {list(incomplete_data.columns)}")

    # Fit the scaler on the complete_data
    scaler = MinMaxScaler()
    scaler.fit(complete_data)

    # Scale the complete_data
    complete_data_scaled = pd.DataFrame(scaler.transform(complete_data), columns=complete_data.columns)

    # Scale the incomplete_data while preserving NaNs
    incomplete_data_scaled = incomplete_data.copy()

    # Scale the incomplete data by temporarily filling NaNs (this is only for scaling)
    transformed_incomplete_data = pd.DataFrame(scaler.transform(incomplete_data.fillna(0)), columns=incomplete_data.columns)

    # Apply the mask to put scaled values back in non-NaN locations
    non_nan_mask = incomplete_data.notna()  # Mask to detect where values are not NaN
    incomplete_data_scaled[non_nan_mask] = transformed_incomplete_data[non_nan_mask]

    return complete_data_scaled, incomplete_data_scaled, complete_data_original, scaler


