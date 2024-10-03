import numpy as np
import matplotlib.pyplot as plt
import os
from sensor_fusion import get_gps_data, get_vio_data


class KalmanFilter:
    '''
    Kalman filter class
    '''
    def __init__(self):
        self.dim_state = 4 # process model dimension
        self.dt = 1 # time increment
        self.q= 0.5 # process noise variable for Kalman filter Q

    def A(self):
        # system/state transition matrix
        dt = self.dt
        return np.matrix([[1,0,1,1], 
                          [0,1,0,1],
                          [0,0,1,0],
                          [0,0,0,1]])

    def Q(self):
        # Q noise follwing Gaussian distribution
        q = self.q
        dt = self.dt
        q1 = ((dt**3)/3) * q 
        q2 = ((dt**2)/2) * q 
        q3 = dt * q 
        return np.matrix([[q1, 0, q2, 0],
                        [0, q1, 0, q2],
                        [q2, 0, q3, 0],
                        [0, q2, 0,  q3]])
        
    def C(self):
        # measurement matrix C, true state space
        return np.matrix([[1,0,0,0],
                         [0,1,0,0]])
    
    def predict(self, x, P):
        # predict state and estimation error covariance to next timestep
        A = self.A()
        x = A*x # state prediction
        P = A*P*A.transpose() + self.Q() # covariance prediction
        return x, P

    def update(self, x, P, z, R):
        # update state and covariance with associated measurement
        C = self.C() 
        gamma = z - C*x #prefit residual
        S = C*P*C.transpose() + R 
        K = P*C.transpose()*np.linalg.inv(S) # optimal Kalman gain
        x = x + K*gamma # update
        I = np.identity(self.dim_state)
        P = (I - K*C) * P #update estimate covariance
        return x, P     
    

def run_filter(gps_data, vio_data, noise_gps, noise_vio):
    # Initial data of x, y
    kf_data = [[0, 0]] 

    # initial state (4x1)
    x = np.array([[0],[0],[0],[0]])

    # initial covariance
    P = np.eye(4)

    # Q - process noise (4x4)
    Q_gps = np.eye(4)
    Q_odom = np.eye(4)

    # R - measurement noise (2x2) -> observation noise assumed as gaussian
    R_gps = noise_gps * np.eye(2)
    R_odom = noise_vio * np.eye(2)

    # filter instance
    filter = KalmanFilter()

    for i in range(gps_data.shape[0]):
        
        # GPS Measurement 
        gps_k = np.array([ [gps_data[i][0]], [gps_data[i][1]]  ])
        odom_k = np.array([ [vio_data[i][0]], [vio_data[i][1]]  ])
        
        # Measurement update
        x, P = filter.update(x, P, gps_k, R_gps)
        x, P = filter.update(x, P, odom_k, R_odom)

        kf_data.append([np.array(x).T[0][0], np.array(x).T[0][1]])

        # predict to next time step
        x, P = filter.predict(x, P)
        
    return kf_data

def main():  
    gps = get_gps_data()
    vio = get_vio_data()
    kf_data = run_filter(gps, vio, 2, 0.3)
    kf_data_array = np.array(kf_data)
    
    plt.figure(figsize=(10,8))
    plt.plot(gps[:,0], gps[:, 1], label="GPS_XY")
    plt.plot(vio[:,0], vio[:, 1], label="VIO")
    plt.plot(kf_data_array[:,0], kf_data_array[:,1], label="KF")
    plt.grid(True)
    plt.title("Kalman Filter vs Raw Data")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xticks(np.arange(-10,40,2))
    plt.yticks(np.arange(-30,2,2))
    plt.legend(loc ="lower left")
    plt.show()

if __name__ == "__main__":
    main()