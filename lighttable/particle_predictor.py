#TODO write kalman filter in here to predict position of particle accounting for uncertainity

#CHATGPT's example of kalman filter
# import numpy as np

# class KalmanFilter:
#     def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise, measurement_matrix, transition_matrix):
#         self.state = initial_state
#         self.covariance = initial_covariance
#         self.process_noise = process_noise
#         self.measurement_noise = measurement_noise
#         self.measurement_matrix = measurement_matrix
#         self.transition_matrix = transition_matrix
    
#     def predict(self):
#         # Predict the next state
#         self.state = np.dot(self.transition_matrix, self.state)
#         # Predict the next covariance
#         self.covariance = np.dot(np.dot(self.transition_matrix, self.covariance), self.transition_matrix.T) + self.process_noise
    
#     def correct(self, measurement):
#         # Calculate the Kalman gain
#         kalman_gain = np.dot(np.dot(self.covariance, self.measurement_matrix.T), np.linalg.inv(np.dot(np.dot(self.measurement_matrix, self.covariance), self.measurement_matrix.T) + self.measurement_noise))
#         # Update the state
#         self.state += np.dot(kalman_gain, (measurement - np.dot(self.measurement_matrix, self.state)))
#         # Update the covariance
#         self.covariance = np.dot((np.eye(len(self.state)) - np.dot(kalman_gain, self.measurement_matrix)), self.covariance)

# # Example usage
# initial_state = np.array([0, 0, 0, 0])  # Initial position and velocity
# initial_covariance = np.eye(4)  # Initial uncertainty
# process_noise = np.eye(4) * 0.01  # Process noise
# measurement_noise = np.eye(2) * 0.1  # Measurement noise
# measurement_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Measurement matrix for position
# transition_matrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])  # Transition matrix for constant velocity motion

# kalman_filter = KalmanFilter(initial_state, initial_covariance, process_noise, measurement_noise, measurement_matrix, transition_matrix)

# # Assuming 'measurements' is a list of particle positions in each image
# for measurement in measurements:
#     kalman_filter.predict()
#     kalman_filter.correct(measurement)

# predicted_positions = kalman_filter.state[:2]  # Extracting predicted positions
