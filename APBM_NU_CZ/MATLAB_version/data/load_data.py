import numpy as np
import torch
import pandas as pd
from pathlib import Path

# Load data
dataFileName = "dataset/decimated_r0_Ttest3000.pt"
dataPath = Path(__file__).resolve().parent.parent.parent / dataFileName
print("Reading data from", dataPath, "......")
[train_input, train_target, cv_input_long, cv_target_long, test_input, test_target] = torch.load(dataPath)

test_flag = False
train_flag = True

if test_flag:
    # save measurements data separately for test data
    for i in range(test_input.size(0)):
        sequence = test_input[i]  # Get the i-th sequence of shape (3, 3000)
        sequence_np = sequence.numpy()  # Convert to NumPy array

        # Save to CSV
        df = pd.DataFrame(sequence_np)
        df.to_csv(f'test_y_{i+1}.csv', header=False, index=False)

    # save state data separately for test data
    for i in range(test_target.size(0)):
        sequence = test_target[i]  # Get the i-th sequence of shape (3, 3000)
        sequence_np = sequence.numpy()  # Convert to NumPy array

        # Save to CSV
        df = pd.DataFrame(sequence_np)
        df.to_csv(f'test_x_{i+1}.csv', header=False, index=False)

if train_flag:
    # save measurements data separately for train data
    for i in range(train_input.size(0)):
        sequence = train_input[i]  # Get the i-th sequence of shape (3, 3000)
        sequence_np = sequence.numpy()  # Convert to NumPy array

        # Save to CSV
        df = pd.DataFrame(sequence_np)
        df.to_csv(f'train_y_{i+1}.csv', header=False, index=False)

    # save state data separately for train data
    for i in range(train_target.size(0)):
        sequence = train_target[i]  # Get the i-th sequence of shape (3, 3000)
        sequence_np = sequence.numpy()  # Convert to NumPy array

        # Save to CSV
        df = pd.DataFrame(sequence_np)
        df.to_csv(f'train_x_{i+1}.csv', header=False, index=False)