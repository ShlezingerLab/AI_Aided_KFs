# DANSE: Data-driven Nonlinear State Estimation (DANSE)

## Paper:
A. Ghosh, A. Honoré and S. Chatterjee, "DANSE: Data-Driven Non-Linear State Estimation of Model-Free Process in Unsupervised Learning Setup," in IEEE Transactions on Signal Processing, vol. 72, pp. 1824-1838, 2024. ([link](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10485649)) 

## Original repository: 
Link: [https://github.com/anubhabghosh/danse_jrnl](https://github.com/anubhabghosh/danse_jrnl) 

## Running the script:

To run the DANSE script, you need to edit the script `run_main_danse.sh` (particularly the variable `mode="train"` (Train) or `mode="test"` (Test))
```
sh run_main_danse.sh
```

## Dependencies 
It is recommended to build an environment either in [`pip`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) or [`conda`](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) and install the following packages (I used `conda` as personal preference):
- PyTorch (1.6.0)
- Python (>= 3.7.0) with standard packages as part of an Anaconda installation such as Numpy, Scipy, Matplotlib, etc. The settings for the code were:
    - Numpy (1.20.3)
    - Matplotlib (3.4.3)
    - Scipy (1.7.3)
    - Scikit-learn (1.0.1)

## Code organization
This would be the required organization of files and folders for reproducing results. If certain folders are not present, they should be created at that level.

````
- run_main_danse.sh (shell script to run 'DANSE' method)
- main_danse_opt.py (main function for training 'DANSE' model)
- rtsnet_params.py (parameters for RTSNet implementation: https://github.com/KalmanNet/RTSNet_TSP/tree/master/Simulations/Lorenz_Atractor/data)

- dataset/ (contains stored datasets in .pt files)

- src/ (contains model-related files)
| - danse.py (for training the unsupervised version of DANSE)
| - rnn.py (class definition of the RNN model for DANSE)

- log/ (contains training and evaluation logs, losses in `.json`, `.log` files)
- models/ (contains saved model checkpoints as `.pt` files)

- utils/ (contains helping functions)

- parameters_opt.py (Python file containing relevant parameters for different architectures)

- bin/ (contains data generation files)
| - ssm_models.py (contains the classes for state space models)
| - generate_data.py (contains code for generating training datasets)
