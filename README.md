# rl-imputation
An implementation of a Q-learning based reinforcement learning algorithm for applications in missing data imputations

# RL-Imputation

This repository contains code for running reinforcement learning (RL) based imputation on datasets, specifically using Q-Learning and Deep Q-Learning (DQN) approaches. The bash script provided automates the process of running the RL imputation over different episode counts and dataset IDs.

## Prerequisites

- **Python 3.x**
- **Pip** for managing Python packages
- **Virtual Environment** (optional, but recommended)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```
### 2. Set Up a Python Virtual Environment (Recommended)

```bash
python3 -m venv rl-imputation-env
source rl-imputation-env/bin/activate  # On Linux/MacOS
```
### 3. Install Dependencies
Navigate to the env/ directory and install the required Python packages using requirements.txt:

```bash
pip install -r env/requirements.txt
```
### 4. Running the RL-Imputation Code
A bash script (run_rl_imputation.sh) is provided to automate the execution of the RL imputation across various episode counts.
#### Script Overview
The script iterates over a list of predefined episode counts and dataset IDs. The results of each run are logged into `run_log.txt`.

#### Execute the Script

```bash
./run_rl_imputation.sh
```
Note: Make sure the script has execute permissions. If it doesn't, you can set it using:

```bash
chmod +x run_rl_imputation.sh
```
### 5. Script Parameters
The main Python script (`rl-imputation.py`) accepts several parameters:

    --method: Specifies the RL method to use (qlearning, dqlearning, customdqlearning).
    --episodes: Number of training steps for the RL agent.
    --id: The ID of the UCI dataset to use for imputation.
    --threshold: Threshold for missing data imputation (default is 0.2).
    --incomplete_data: Path to the incomplete data CSV file.
    --complete_data: Path to the complete data CSV file.

### 6. Log Output

All logs and potential error messages are captured in run_log.txt. After running the script, you can review the log file for details on the execution.

``` bash
cat run_log.txt
```
### 7. Results

The imputed datasets and model results are saved in the `results/` directory.
