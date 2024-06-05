#TODO write kalman filter in here to predict position of particle accounting for uncertainity

import numpy as np
import sys

#helping me to understand kalman filter
# you have your current position
#you predict next position with approximately particle size uncertaintiy in all directions
#you measure next position and correct, there should be no uncertainity on next position (at most, uncertainity would be the size of the particle)



#the state matrix should contain position and velocity so according to java code would look like
x_pos = 4
y_pos = 2
x_vel = 10
y_vel = -2

statematrix = np.array([[x_pos],
                        [y_pos],
                        [x_vel],
                        [y_vel]])

statematrix #4x1 matrix

#the evolution matrix is a changed identity matrix and would look like
evolutionmatrix = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

evolutionmatrix #4x4 matrix 

#the measurement matrix should just be large enough to hold the measured data
measurementmatrix = np.array([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,0],
                              [0,0,0,1]])

measurementmatrix #4x4 matrix 

#the state covariance reflects the confidnece in the state measurements. 
#for our purposes we should trust positon measurements completely
#therefore we use the smalled position system float that can be normalised
min_normal_float = sys.float_info.min

statecovariance = np.array([[min_normal_float, 0, 0, 0],
                            [0, min_normal_float, 0, 0],
                            [0, 0, min_normal_float, 0],
                            [0, 0, 0, min_normal_float]])

statecovariance #4x4 matrix

#position noise should be 1/3 max search radius
#velocity noise should be 1/3 max search radius
#therefore the process covariance would be

search_radius = 40

process_noise = (1/3) * search_radius

processcovariancematrix = np.array([[process_noise,0,0,0],
                                   [0,process_noise,0,0],
                                   [0,0,process_noise,0],
                                   [0,0,0,process_noise]])

processcovariancematrix #4x4 matrix

#measurement noise should be 1/10 of the mean particle radius
#therefore the observation covariance would be

mean_particle_radius = 5

observation_noise = (1/10)*mean_particle_radius

observationcovariancematrix = np.array([[observation_noise**2, 0, 0, 0],
                                        [0, observation_noise**2, 0, 0],
                                        [0, 0, observation_noise**2, 0],
                                        [0, 0, 0, observation_noise**2]])

observationcovariancematrix #4x4 matrix



#attempting the prediction steps
#state * evolution
#statecovariance = (evolution * (statecovariance * transposed(evolution))) + process covariance

statematrix
predictedstate = np.dot(evolutionmatrix, statematrix) #4x1 matrix
statecovariance = np.add(np.dot(evolutionmatrix, np.dot(statecovariance, evolutionmatrix.T)), processcovariancematrix) #4x4 matrix


#updating the model steps with measured values
#if the measured matrix is null i.e. the particle isn't linked to another one then
statematrix = predictedstate


x_measured = 13
y_measured = 1
x_vel_measured = 9
y_vel_measured = -1
#else
measuredmatrix = np.array([x_measured, y_measured, x_vel_measured, y_vel_measured])

measuredmatrix = measuredmatrix.reshape(4,1) #4x1 matrix 

#temp matrix = (measurement matrix * (statecovariance * transposed(measurment matrix))) + observationcovariance matrix 
tempmatrix = np.add(np.dot(measurementmatrix, np.dot(statecovariance, measurementmatrix.T)), observationcovariancematrix)

tempmatrix #4x4 matrix 

#kalman gain = (statecovariance * (transposed(measurement matrix)) * inverse(tempmatrix)))
kalmangain = np.dot(statecovariance, np.dot(measurementmatrix.T, np.linalg.inv(tempmatrix)))

kalmangain #4x4 matrix 


#calculating the new state using the predicted state, measured state, and kalman gain
statematrix = np.add(predictedstate, np.dot(kalmangain, np.subtract(measuredmatrix, np.dot(measurementmatrix, predictedstate))))

statematrix #4x1 matrix 

#calculating the covariance matrix using the kalman gain and the previous state covariance matrix
statecovariance = np.dot((np.subtract(np.identity(4), np.dot(kalmangain, measurementmatrix))), statecovariance)
statecovariance


#calculating position and velocity error from the state covariance
PositionError = np.sqrt((statecovariance[0,0] + statecovariance[1,1]) / 2)
PositionError

VelocityError = np.sqrt((statecovariance[2,2] + statecovariance[3,3]) / 2)
VelocityError

H = measurementmatrix
P = statecovariance
R = observationcovariancematrix
Xp = predictedstate
K = kalmangain
Xm = measuredmatrix

Xm.shape