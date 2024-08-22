import numpy as np
import torch
import pandas as pd

# load data
DatafileName = 'decimated_r0_Ttest3000.pt'
[train_input, train_target, cv_input_long, cv_target_long, test_input, test_target] = torch.load(DatafileName)

# save measurements data separately
for i in range(test_input.size(0)):
    sequence = test_input[i]  # Get the i-th sequence of shape (3, 3000)
    sequence_np = sequence.numpy()  # Convert to NumPy array

    # Save to CSV
    df = pd.DataFrame(sequence_np)
    df.to_csv(f'test_y_{i+1}.csv', header=False, index=False)

# save state data separately
for i in range(test_target.size(0)):
    sequence = test_target[i]  # Get the i-th sequence of shape (3, 3000)
    sequence_np = sequence.numpy()  # Convert to NumPy array

    # Save to CSV
    df = pd.DataFrame(sequence_np)
    df.to_csv(f'test_x_{i+1}.csv', header=False, index=False)