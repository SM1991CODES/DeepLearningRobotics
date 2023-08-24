"""
Implements a KF filter in 2D to track position and velocity over certain time steps
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

class CarSimulator(object):
    """
    Class implements a simulator for a car whose position and velocity are to be tracked
    Returns true position, measured position and true velocity at each simulation time-step
    """

    def __init__(self, start_pos, start_vel, t_step_s, max_pos_error=1):
        """
        Default constructor- set's up some parameters for the simulator
        """
        
        self.pos = start_pos
        self.vel = start_vel
        self.t_step = t_step_s
        self.max_error_pos = max_pos_error
        print("====== CarSim initialized ============")
    
    def get_sim_data(self):
        """
        Function returns data for next simulation time step data
        Time step is fixed in __init__()
        """
        
        pos_gt = self.pos
        pos_meas = pos_gt + np.random.uniform(-1*self.max_error_pos, self.max_error_pos)  # measure - position with some Gaussian noise
        vel_gt = self.vel
        self.pos += self.vel * self.t_step  # update for next time step

        return (pos_gt, pos_meas, vel_gt)


class KalmanFilter2D(object):
    """
    Class implements the 2D Kalman filter to track both the position and velocity
    """

    def __init__(self, X_init_nx1, P_init_nxn, F_nxn, Q_init_nxn, R_nxn, H_1xn) -> None:
        """
        Default constructor - set's up filter parameters
        """

        self.X_nx1 = X_init_nx1  # state mean
        self.P_nxn = P_init_nxn  # state cov.
        self.F_nxn = F_nxn  # state transition
        self.Q_nxn = Q_init_nxn  # process noise
        self.H_1xn = H_1xn  # meas. function
        self.R_nxn = R_nxn  # meas noise
        self.K = 0  # Kalman gain
    
    def print_filter_params(self):
        """
        Helper function to print all matrices and states of the KF
        """
        print("State mean (X)-> ", self.X_nx1)
        print("State cov (P)-> ", self.P_nxn)
        print("State transition (F)-> ", self.F_nxn)
        print("Process noise cov. (Q)-> ", self.Q_nxn)
        print("Kalman Gain (K) -> ", self.K)
        print("Measurement noise cov. (R)-> ", self.R_nxn)
    
    def predict(self):
        """
        Predict next state and cov. based on previous state and motion model/ state transition matrix
        """

        self.X_nx1 = np.dot(self.F_nxn, self.X_nx1) + 0  # we have considered u = 0 here
        self.P_nxn = np.dot(self.F_nxn, np.dot(self.P_nxn, self.F_nxn.T)) + self.Q_nxn
        return self.X_nx1, self.P_nxn

    def update(self, meas_pos):
        """
        Update predicted states and covariances using new measurements
        """
        

        self.S = np.dot( np.dot(self.H_1xn, self.P_nxn), self.H_1xn.T ) + self.R_nxn  # HPH' + R
        self.K = np.dot( np.dot(self.P_nxn, self.H_1xn.T), inv(self.S) )
        y = meas_pos - np.dot(self.H_1xn, self.X_nx1)
        self.X_nx1 += np.dot(self.K, y)
        self.P_nxn = self.P_nxn - np.dot( np.dot(self.K, self.H_1xn), self.P_nxn)

        return self.X_nx1, self.P_nxn




if __name__ == "__main__":

    car_sim = CarSimulator(start_pos=10, start_vel=2, t_step_s=0.5, max_pos_error=4)

    X_init_nx1 = np.array([[-25.0, 1.0]]).T  # state means
    P_init_nxn = np.array([[400., 0],
                           [0, 25.]])  # we expect state to be wrong by as much as +-10 and vel. by +- 5
    
    dt = 0.5  # same as used in simulation
    F_nxn = np.array([[1., dt],
                      [0, 1.]])
    
    Q_init_nxn = np.array([[0.05, 0.],
                           [0., 0.01]])  # small process noise
    
    H_1xn = np.array([[1., 0.]])  # meas. matrix/ function -> maps state to meas

    R_nxn = np.array([[9.]])  # we measure position only
    
    kalman_filter = KalmanFilter2D(X_init_nx1=X_init_nx1, P_init_nxn=P_init_nxn, F_nxn=F_nxn, Q_init_nxn=Q_init_nxn, 
                                   R_nxn=R_nxn, H_1xn=H_1xn)
    
    kalman_filter.print_filter_params()

    gt_pos = []
    meas_pos = []
    gt_vel = []
    kf_pos = []
    kf_vel = []
    for i in range(100):
        pos_gt, pos_meas, vel_gt = car_sim.get_sim_data()

        pos_vel, cov = kalman_filter.predict()
        print("Covariance predict -> ", cov)
        pos_vel, cov = kalman_filter.update(pos_meas)
        print("Covariance update -> ", cov)

        gt_pos.append(pos_gt)
        meas_pos.append(pos_meas)
        gt_vel.append(vel_gt)
        kf_pos.append(pos_vel[0])
        kf_vel.append(pos_vel[1])

    plt.plot(gt_pos, "g-")
    plt.plot(kf_pos, "c--")
    plt.plot(meas_pos, "b*")
    plt.plot(gt_vel, "r")
    plt.show()
    

