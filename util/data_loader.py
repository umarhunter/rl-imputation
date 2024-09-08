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
    dataset = get_data(datasetid)
    df = dataset.data.original
    missing_df = generate_missing_df(df, missing_rate)

    missing_values_count = missing_df.isna().sum().sum()

    if datasetid == 17:
        target_column = ['ID', 'Diagnosis']  # Hardcoded for the Breast Cancer Wisconsin dataset
        incomplete_data = missing_df.drop(columns=target_column)
        complete_data = df.drop(columns=target_column)
        return complete_data, incomplete_data

    # Output result
    if missing_values_count > 0:
        logging.info(f"The DataFrame contains {missing_values_count} missing values after load_dataset()")

def preprocess_data(complete_data, incomplete_data):
    # Ensure complete_data is filled for fitting the scaler
    complete_data_filled = complete_data.fillna(0)

    # Fit the scaler on the complete data
    scaler = MinMaxScaler()
    scaler.fit(complete_data_filled)

    # Scale both complete_data and incomplete_data
    complete_data_scaled = pd.DataFrame(scaler.transform(complete_data_filled), columns=complete_data.columns)

    # IMPORTANT: Do not fill missing values in incomplete_data
    incomplete_data_scaled = pd.DataFrame(scaler.transform(incomplete_data), columns=incomplete_data.columns)

    return complete_data_scaled, incomplete_data_scaled, scaler
