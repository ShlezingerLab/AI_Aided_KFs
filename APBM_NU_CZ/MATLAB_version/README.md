# Augmented Physics-Based Model (APBM) in MATLAB

APBM is designed to compensate for discrepancies between the true dynamics of a system and an approximate or partially-known model.

This folder contains scripts for APBM training. The two main scripts included are:
- `lorenz_attractor_online_unsupervised.m`
- `lorenz_attractor_offline_unsupervised.m`

Other scripts in this folder are auxiliary functions and test files.

The unsupervised version is written in MATLAB because nonlinear filtering toolboxes, such as EKF, UKF, and CKF, are easy to implement and efficient to process in this environment.

## Implementation Details

### `lorenz_attractor_online_unsupervised.m`
- Implements the standard APBM using the Cubature Kalman Filter (CKF).
- Input: 10 sequences of noisy measurements, each of size (3 × 3000).
- True state: 10 sequences of true states, each of size (3 × 3000), used for error computation.

### `lorenz_attractor_offline_unsupervised.m`
- Uses 100 sequences of noisy measurements, each of size (3 × 3000), for offline training.
- The neural network (NN) parameters are continuously trained over the 100 datasets. States are re-initialized at the beginning of training on each new dataset.
- After filtering through the training datasets, the NN parameters are saved and used for test data.
- During testing, the APBM can either remain fixed or continue evolving.

## Attention

- The `Sensor Fusion and Tracking Toolbox` is required for the CKF implementation.
- In the training process, the `train_flag` is set to `0/False` by default, which loads the pre-trained NN parameters from `nn_parameter_offline_unsupervised.mat`. If you want to train your own NN, set `train_flag` to `1/True` and verify the training settings accordingly.

