#!/bin/bash

# Define the episode counts and dataset IDs
#EPISODE_COUNTS=(50 100 1000 2000 5000 10000 20000 25000 50000 100000)
EPISODE_COUNTS=(50 100 1000 2000 5000 10000 20000 25000)
# DATASET_IDS=(94 59 17 332 350 189 484 149)
DATASET_IDS=(17)

LOG_FILE="run_log.txt"

# Clear the log file
echo "Log for RL Imputation Run" > $LOG_FILE

# Run the script for each dataset ID and each episode count
for DATASET_ID in "${DATASET_IDS[@]}"
do
    for EPISODES in "${EPISODE_COUNTS[@]}"
    do
        echo "Running dataset ID $DATASET_ID with $EPISODES episodes"
        python rl-imputation.py --method="dqlearning" --episodes="$EPISODES" --id="$DATASET_ID" 2>> $LOG_FILE || {
            echo "Error encountered with dataset ID $DATASET_ID and $EPISODES episodes" >> $LOG_FILE
        }
    done
done

echo "Finished all tasks. Check $LOG_FILE for any errors."
