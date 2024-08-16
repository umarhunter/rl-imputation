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


def get_dataset_name(dataset_id):
    dataset_names = {
        94: "spambase",
        59: "letter_recognition",
        17: "breast_cancer_wisconsin",
        332: "online_news_popularity",
        350: "default_credit_card_clients",
        189: "parkinsons_telemonitoring",
        484: "travel_reviews",
        149: "statlog_vehicle_silhouettes"
    }
    return dataset_names.get(dataset_id, "unknown_dataset")


def load_dataset(datasetid, missing_rate=0.1):
    dataset = get_data(datasetid)
    df = dataset.data.original
    missing_df = generate_missing_df(df, missing_rate)

    missing_values_count = missing_df.isna().sum().sum()
    # Output result
    if missing_values_count > 0:
        logging.info(f"The DataFrame contains {missing_values_count} missing values after load_dataset()")

    return df, missing_df


def preprocess_data(incomplete_data, complete_data):
    # Identify categorical and numerical columns
    categorical_columns = incomplete_data.select_dtypes(include=['object', 'category']).columns
    numerical_columns = incomplete_data.select_dtypes(include=['number']).columns

    # Initialize scalers and encoders
    scalers = {col: MinMaxScaler() for col in numerical_columns}
    encoders = {col: LabelEncoder() for col in categorical_columns}

    # Encode categorical variables
    for col in categorical_columns:
        # Fit on combined data to handle all possible categories
        combined_data = pd.concat([incomplete_data[col], complete_data[col]], axis=0)
        combined_filled = combined_data.fillna('MISSING_PLACEHOLDER').astype(str)

        encoders[col].fit(combined_filled)

        # Transform both datasets
        incomplete_filled = incomplete_data[col].fillna('MISSING_PLACEHOLDER').astype(str)
        complete_filled = complete_data[col].fillna('MISSING_PLACEHOLDER').astype(str)

        incomplete_data[col] = encoders[col].transform(incomplete_filled)
        complete_data[col] = encoders[col].transform(complete_filled)

        # Restore NaNs in incomplete data
        placeholder_index = encoders[col].transform(['MISSING_PLACEHOLDER'])[0]
        incomplete_data.loc[incomplete_data[col] == placeholder_index, col] = np.nan

    # Scale numerical variables
    for col in numerical_columns:
        # Fit scaler on complete data to ensure consistent scaling
        scalers[col].fit(complete_data[[col]])

        # Transform both datasets
        incomplete_data[col] = scalers[col].transform(incomplete_data[[col]])
        complete_data[col] = scalers[col].transform(complete_data[[col]])

    return incomplete_data, complete_data
