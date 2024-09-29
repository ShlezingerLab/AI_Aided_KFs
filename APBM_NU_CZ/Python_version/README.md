# Augmented Physics-Based Model (APBM) in Python

APBM is designed to compensate for discrepancies between the true dynamics of a system and an approximate or partially-known model.

This folder contains scripts for APBM training. The three main scripts included are:
- `lorenz_attractor_online_unsupervised.py`
- `lorenz_attractor_offline_supervised_fcnn.py`
- `lorenz_attractor_offline_supervised_mlp.py`

Other scripts in this folder are auxiliary functions and test files.

The supervised version is written in Python because neural network (NN) training (including 'dataloader', 'loss', and 'optimizer') is easy to implement with PyTorch.

## Implementation Details

### `lorenz_attractor_online_unsupervised.py`
- Implements the standard APBM using the Cubature Kalman Filter (CKF), corresponding to the MATLAB script `lorenz_attractor_online_unsupervised.m`.
- Input: 10 sequences of noisy measurements, each of size (3 × 3000).
- True state: 10 sequences of true states, each of size (3 × 3000), used for error computation.

### `lorenz_attractor_offline_supervised_fcnn.py`
- Uses a sequence of true states, each of size (3 × 3000), for offline training.
- Input: 3 × 2999 states as \(x_k\) and 3 × 2999 states as \(x_{k-1}\).
- After filtering through the training datasets, the NN parameters are saved and used for test data.
- During testing, the APBM remains fixed in the dynamics model of the system.

### `lorenz_attractor_offline_supervised_mlp.py`
- Similar to the above, but the built-in fully connected NN from `nn.Sequential` is replaced with a self-defined MLP in `tmlp.py`.
- This offers more flexibility for tuning parameters and monitoring the training process.

## Attention

- Required libraries: `numpy`, `torch`, `pandas`, `filterpy`.
- The `filterpy` library is specifically used for the Cubature Kalman Filter (CKF) implementation. For more details and installation instructions, visit the [filterpy GitHub repository](http://github.com/rlabbe/filterpy).
- During training, the `train_flag` is set to `False` by default, loading the pre-trained NN parameters from `fcnn_parameter_shuffle.pt` or `mlp_parameter.pt`. To train your own NN, set `train_flag` to `True` and ensure the training settings are properly configured.
