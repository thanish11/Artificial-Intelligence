import numpy as np
import matplotlib.pyplot as plt
def kalman_filter(measurements, initial_state, initial_covariance, process_noise, measurement_noise):
 # Kalman filter parameters
 A = np.array([[1]]) # State transition matrix (constant velocity model)
 H = np.array([[1]]) # Observation matrix
 Q = np.array([[process_noise]]) # Process noise covariance
 R = np.array([[measurement_noise]]) # Measurement noise covariance
 # Initialize state and covariance
 x_hat = initial_state
 P = initial_covariance
 # Lists to store filtered states and measurements
 filtered_states = []
 filtered_measurements = []
 for z in measurements:
    # Prediction step
    x_hat_minus = np.dot(A, x_hat)
    P_minus = np.dot(np.dot(A, P), A.T) + Q
    # Update step
    K = np.dot(np.dot(P_minus, H.T), np.linalg.inv(np.dot(np.dot(H, P_minus), H.T) + R))
    x_hat = x_hat_minus + np.dot(K, (z - np.dot(H, x_hat_minus)))
    P = np.dot((np.eye(len(x_hat)) - np.dot(K, H)), P_minus)
    #Store filtered states and measurements
    filtered_states.append(x_hat[0, 0])
    filtered_measurements.append(z)
 return filtered_states, filtered_measurements
# Simulate a one-dimensional constant velocity system
np.random.seed(42)
true_state = [0]
for _ in range(99):
 true_state.append(float(np.random.normal(true_state[-1], 0.5, 1)))
noise = np.random.normal(0, 0.06, 100)
measurements = true_state + noise
print(f"True state: \n{true_state[:10]}")
print(f"noise: \n{noise[:10]}")
print(f"measurements: \n{measurements[:10]}")
# Initial state estimate and covariance
initial_state_estimate, initial_covariance = np.array([[0]]), np.array([[1]])
# Process and measurement noise parameters
process_noise = 0.1
measurement_noise = 0.5
# Run the Kalman filter
filtered_states, filtered_measurements = kalman_filter(measurements, initial_state_estimate,
 initial_covariance, process_noise, measurement_noise)
print(f"filtered states: \n{filtered_states[:10]}")
# Plot the results 
plt.figure(figsize=(16, 6))
plt.plot(true_state, label='True State', linestyle='dashed', color='blue')
plt.scatter(range(len(measurements)), measurements, label='Measurements', color='orange', 
marker='x', s=20)
plt.plot(filtered_states, label='Filtered State', color='green', linewidth=2)
plt.title('Kalman Filter Example')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
print(f"The resultant position of the agent is (100, {filtered_states[-1]:.4f})")
