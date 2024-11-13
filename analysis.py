import os
import json
import numpy as np


def load_json_files(directory):
    """Load all JSON files in a directory and return their contents as a list of dictionaries."""
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                data.append(json.load(f))
    return data


def calculate_performance_metrics(data):
    """Calculate performance metrics from a list of experiment results."""
    best_individual = None
    best_mae = float('inf')
    best_mse = float('inf')
    best_tolerance_match_rate = 0
    best_mean_reward = float('-inf')

    total_mae = 0
    total_mse = 0
    total_tolerance_match_rate = 0
    total_mean_reward = 0
    num_files = len(data)

    for experiment in data:
        mae = experiment["MAE"]
        mse = experiment["MSE"]
        tolerance_match_rate = experiment["tolerance_match_rate"]
        mean_reward = experiment.get("mean_reward", 0)

        total_mae += mae
        total_mse += mse
        total_tolerance_match_rate += tolerance_match_rate
        total_mean_reward += mean_reward

        # Track the best individual performance based on MAE, MSE, or tolerance_match_rate
        if mae < best_mae:
            best_mae = mae
            best_individual = experiment

        if mse < best_mse:
            best_mse = mse
            best_individual = experiment

        if tolerance_match_rate > best_tolerance_match_rate:
            best_tolerance_match_rate = tolerance_match_rate
            best_individual = experiment

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_individual = experiment

    # Calculate average performance for the folder
    avg_mae = total_mae / num_files
    avg_mse = total_mse / num_files
    avg_tolerance_match_rate = total_tolerance_match_rate / num_files
    avg_mean_reward = total_mean_reward / num_files

    return {
        "best_individual": best_individual,
        "avg_mae": avg_mae,
        "avg_mse": avg_mse,
        "avg_tolerance_match_rate": avg_tolerance_match_rate,
        "avg_mean_reward": avg_mean_reward
    }


def compare_all_test_folders(results_dir):
    """Compare the performance of all test folders in the results directory."""
    test_folders = sorted(
        [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d.startswith('test_')])

    best_overall_folder = None
    best_overall_performance = float('inf')
    overall_results = {}

    for folder in test_folders:
        folder_path = os.path.join(results_dir, folder)
        folder_data = load_json_files(folder_path)
        folder_metrics = calculate_performance_metrics(folder_data)
        overall_results[folder] = folder_metrics

        # Determine the best overall folder based on average MAE
        if folder_metrics["avg_mae"] < best_overall_performance:
            best_overall_performance = folder_metrics["avg_mae"]
            best_overall_folder = folder

    return {
        "overall_results": overall_results,
        "best_overall_folder": best_overall_folder
    }



results_directory = "results"
final_results = compare_all_test_folders(results_directory)

# Output the best overall folder and its metrics
print(f"Best overall folder: {final_results['best_overall_folder']}")
for folder, metrics in final_results['overall_results'].items():
    print(f"\nMetrics for {folder}:")
    print(f"Best Individual Performance:")
    print(f"  Dataset: {metrics['best_individual']['dataset_name']}")
    print(f"  MAE: {metrics['best_individual']['MAE']}")
    print(f"  MSE: {metrics['best_individual']['MSE']}")
    print(f"  Tolerance Match Rate: {metrics['best_individual']['tolerance_match_rate']:.2f}%")
    print(f"  Mean Reward: {metrics['best_individual']['mean_reward']}")
    print(f"Average Metrics:")
    print(f"  Average MAE: {metrics['avg_mae']}")
    print(f"  Average MSE: {metrics['avg_mse']}")
    print(f"  Average Tolerance Match Rate: {metrics['avg_tolerance_match_rate']:.2f}%")
    print(f"  Average Mean Reward: {metrics['avg_mean_reward']}")
