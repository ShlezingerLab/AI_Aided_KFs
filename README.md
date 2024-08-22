# AI-Aided Kalman Filters

This repository contains the implementation and comparison of AI-aided Kalman filter techniques applied to measurements observed from a Lorenz Attractor system. The purpose of this project is to explore how artificial intelligence can enhance traditional Kalman filter methods, improving accuracy and robustness in nonlinear state estimation.

## Algorithms Included

The following filters and techniques are implemented in this repository:

- **Extended Kalman Filter (EKF)** 
- **Cubature Kalman Filter (CKF)** 
- **Particle Filter (PF)** 
- **KalmanNet:** an interpretable, low complexity, and data-efficient DNN-aided real-time state estimator by learning the Kalman gain.
- **RTSNet:** an iterative hybrid model-based/data-driven algorithm for smoothing in dynamical systems.
- **Data-driven Nonlinear State Estimation (DANSE):** a data-driven nonlinear state estimation method.
- **Augmented Physics-based Model (APBM):** a model that combines physical modeling with data-driven methods for enhanced state estimation.

## Project Structure

The repository is organized as follows:

- `dataset/`: the common dataset of the observations and ground truth.
- `figs/`: simulation and experiment results.
- `RTSNet_IL/, DANSE_KTH, APBM_NU_CZ`: the codes of implementation for the algorithm indicated by the folder name.

## Getting Started

To get started with this project, clone the repository and follow the instructions in each subfolder. Please pay attention to the environment and package requirements.

## Contributions

This work is a collaborative effort by researchers from:

- Ben-Gurion University
- ETH Zürich
- KTH Royal Institute of Technology
- University of West Bohemia
- Northeastern University

## Citation

For more details or to cite this work, plaese refer to the paper:
	```
     TBD
    ```
	

## License

TBD.

## Contact

TBD.


