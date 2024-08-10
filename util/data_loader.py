from ucimlrepo import fetch_ucirepo, list_available_datasets
import util


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


def load_dataset(datasetid, missing_rate=0.2):
    dataset = get_data(datasetid)
    df = dataset.data.original
    missing_df = util.generate_missing_df(df, missing_rate)
    return df, missing_df
