# Kalman Filter and Particle Filter Implementations

This repository contains custom implementations of the Kalman Filter and Particle Filter algorithms. Both filters are essential for state estimation in dynamic systems, particularly in robotics and computer vision applications for a localization tasks.

### Table of Contents
- [Kalman Filter](#kalman-filter)
  - [Overview](#overview)
  - [Key Concepts](#key-concepts)
  - [Algorithm](#algorithm)
  - [Implementation](#implementation)
  - [Usage](#usage)
- [Particle Filter](#particle-filter)
  - [Overview](#overview-1)
  - [Key Concepts](#key-concepts-1)
  - [Algorithm](#algorithm-1)
  - [Implementation](#implementation-1)
  - [Usage](#usage-1)

## Kalman Filter

### Overview

The Kalman Filter is an efficient recursive filter that estimates the state of a linear dynamic system from a series of noisy measurements. It provides a means to update the state estimate based on new measurements over time.

### Key Concepts

- **State Vector**: Represents the current state of the system.
- **Measurement Vector**: Represents the observed data.
- **Prediction Step**: Uses the system model to predict the next state.
- **Update Step**: Combines the prediction with the new measurements to improve the estimate.

### Algorithm

The Kalman Filter algorithm can be summarized in two main steps:

1. **Prediction**:
   - Predict the state: 
   $$x_{k} = A_k {x}_{k-1}$$
   

   - Some times we have control input $u_k$ and process noise $w_k$ from gaussian $Q$, then the prediction step will be:
   
   $$x_{k} = A_k {x}_{k-1} + B_k u_k + w_k$$

   - Predict the error covariance: $$P_{k} = A_k P_{k-1} A_k^T + Q_k$$

2. **Update**:
   - Compute the Kalman Gain: $$K_k = P_{k} C_k^T (C_k P_{k} C_k^T + R_k)^{-1}$$

   - Update the state estimate: $$x_{k} = {x}_{k} + K_k (z_k - C_k {x}_{k})$$
   - Update the error covariance: $$P_{k} = (I - K_k C_k) P_{k}$$

### Implementation

#### Class Structure

- **KalmanFilter**: Contains methods for initializing the filter, predicting the next state, and updating the state with measurements. With default, $A$, $Q$, $C$ matrices initialize. Can be modified by user.

#### Main Functions

- **init()**: Initializes the state and covariance matrices.
- **predict()**: Predicts the next state and updates the state covariance.
- **update()**: Updates the state estimate and covariance based on measurements.
- **run_filter()**: Combines the predict and update steps for a series of measurements.

### Usage
```python
from kalman_filter import KalmanFilter

kf = KalmanFilter()
kf.predict(x, P) # x is the state vector, P is the covariance matrix
kf.update(x, P, gps_data_arr, odom_data_arr) #sample data
```
or 
```python
from kalman_filter import KalmanFilter

kf = KalmanFilter()
kf.run_filter(gps_data_arr, odom_data_arr, gps_noise, odom_noise)
```

## Particle Filter

### Overview

The Particle Filter is utilized in Monte Carlo Localization method for implementing a recursive Bayesian filter. It is particularly useful for estimating the state of a system that is non-linear and/or especially, non-Gaussian. 

### Key Concepts

- **Particles**: Each particle represents a possible state of the system.
- **Weights**: Each particle has an associated weight, indicating its likelihood given the observed data.
- **Resampling**: A process to select particles based on their weights to focus on the most likely states.

### Algorithm

The Particle Filter algorithm can be summarized as follows:

1. **Initialization**:
   - Generate $N$ particles ${x_0^i}$ sampled from the initial state distribution $p(x_0 | z_0)$.
   - Initialize weights $w_0^i = \frac{1}{N}$ for each particle.

2. **Prediction**:
   - For each particle $i$, propagate the state:
   $${x_k^i} \sim p(x_{k} | x_{k-1}^i, u_{k})$$
   where $u_k$ is the control input.

3. **Update**:
   - For each particle $i$, update the weight based on the observation:
   $${w_k^i} \propto w_{k-1}^i \cdot p(z_k | x_k^i)$$
   - Normalize the weights:
   $$w_k^i = \frac{w_k^i}{\sum_{j=1}^{N} w_k^j}$$

4. **Resampling**:
   - Resample $N$ particles based on their weights $w_k^i$ to create a new set of particles ${x_k^{i'}}$

##### Intution: As we iteratively resample, the sample set gets smaller and smaller until we reach a certain point with high probability of the robot being there.

### Implementation

#### Class Structure

- **Particle**: Represents an individual particle with state and weight attributes.
- **ParticleFilter**: Manages the collection of particles and implements the filtering algorithm.

#### Main Functions

- **init()**: Initializes the particles based on the initial state.
- **predict()**: Propagates the particles based on the system dynamics.
- **update()**: Updates the weights of the particles based on the measurements.
- **resample()**: Resamples the particles to focus on the most likely states.

### Usage

```python
from particle_filter import ParticleFilter

pf = ParticleFilter()
pf.predict()
pf.update(data)
pf.resample()
```
