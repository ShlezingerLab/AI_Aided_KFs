### KalmanNet Description for GitHub README:

**KalmanNet**: Leveraging the structure of the Kalman Filter within a neural network framework, KalmanNet serves as a state estimator for dynamical systems with partially known or non-linear dynamics. This hybrid model blends the interpretability and efficiency of traditional Kalman Filters with the flexibility of deep learning, particularly Recurrent Neural Networks (RNNs). It operates effectively in real-time, adapting to complex and non-linear systems by learning from data to optimize state estimation tasks, even when complete system dynamics are not available. Ideal for applications where model uncertainties or non-linearities exist, KalmanNet provides robust performance enhancements over traditional methods.

### Lorentz Attractor Experiment:

The Lorentz Attractor experiment involves the use of KalmanNet to track the highly non-linear and chaotic behavior typical of the Lorenz attractor system. This system is a set of three-dimensional, non-linear differential equations that produce chaotic solutions, which are sensitive to initial conditions. The experiment leverages KalmanNet’s ability to handle the challenges posed by the Lorenz system’s non-linearity and sensitivity, demonstrating superior tracking and estimation capabilities compared to traditional model-based filters. This is particularly evident in scenarios with model mismatches due to approximations or sampling from continuous-time to discrete-time systems, where KalmanNet effectively learns to mitigate these issues and provides accurate state estimations.

**KalmanNet:**
- Guy Revach, Nir Shlezinger, Ruud J.G. van Sloun, Yonina C. Eldar, "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics," 2022, available in IEEE Xplore. [Access here](https://ieeexplore.ieee.org/document/9733186)

**To run the Lorenz Attractor experiment using EKF or KalmanNet:**

1. Navigate to `RTSNet_batch_version`.
2. Open `main_lor_decimation.py`.
3. Uncomment the section for EKF or KalmanNet, depending on which algorithm you want to run.
4. Ensure that other algorithms, including any versions of RTSNet, are commented out.

**To run the Lorenz Attractor experiment using the Particle Filter (PF):**

1. Go to `RTSNet_original_version`.
2. Open `main_lor_decimation.py`.
3. Comment out all algorithms except for the Particle Filter.

**Data Handling:**
- The data can be regenerated, but it is not necessary for this run. Please ensure the data generation part remains commented out and use the load data function instead.
- The existing data can be found in `Simulations/Lorenz_Atractor/Data`.

**Enjoy running your experiments!**
