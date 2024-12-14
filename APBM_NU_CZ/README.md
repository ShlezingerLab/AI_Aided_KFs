## Augmented Physics-Based Model (APBM)

APBM is designed to compensate for discrepancies between the true dynamics of a system and an approximate or partially-known model.   

In this project, we specifically address the challenge where the first state dynamics of the Lorenz Attractor is missing and considered as a random walk. To overcome this, a data-driven AI component, implemented using a Multiple Layer Perceptron (MLP), is employed to learn and approximate the true dynamics.

### Implementation Details

The APBM is implemented in two versions with different environment requirements:

- **MATLAB:**
  - Version: `>= R2023a`
  - Required Toolboxes:
    - `Sensor Fusion and Tracking Toolbox`
    - `Parallel Computing Toolbox`
  - Training:
	- Offline Unsupervised: Trained by the measuerments of training data (without true state)
	- Online Unsupervised: Trained by the measuerments of test data (without true state)

- **Python:**
  - Version: `>= 3.7`
  - Required Packages:
    - `numpy >= 1.26.4`
    - `torch >= 2.4.0`
    - `pandas >= 2.2.2`
	- `filterpy >= 1.4.5`
  - Training:
	- Offline Supervised: Trained by the true states of training data
	- Online Unsupervised: Trained by the measuerments of test data (without true state)

Both implementations aim to demonstrate how the combination of physics-based models with AI-driven components can effectively estimate the system's true state, even when part of the dynamics is unknown or approximated.

### Attention
The library `filterpy` is for CKF implementation.

The `filterpy` library is used specifically for the implementation of the Cubature Kalman Filter (CKF). For more information and installation instructions, please refer to the [filterpy GitHub repository](http://github.com/rlabbe/filterpy).

