"""
Demostrates a simple 2D linear motion UKF using the built-in filterpy functions
"""

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
import matplotlib.pyplot as plt


class CarSimLin2D(object):
    """
    Class implements a car simulator in 2D - linear motion, constant velocity
    """

    def __init__(self, init_pos_xy, init_vel_xy, del_t_s, max_err_pos_x_y) -> None:
        """
        Default constr. - sets up the simulator
        """

        self.pos_x = init_pos_xy[0]
        self.pos_y = init_pos_xy[1]
        self.vel_x = init_vel_xy[0]
        self.vel_y = init_vel_xy[1]
        self.dt = del_t_s
        self.err_x = max_err_pos_x_y[0]
        self.err_y = max_err_pos_x_y[1]
    
    def get_sim_data(self):
        """
        Advances simulation 1 time step and returns GT and measured position and velocities
        """

        # advance pos and add some noise to measured pos
        self.pos_x  = self.pos_x + self.dt * self.vel_x
        meas_pos_x = self.pos_x + np.random.uniform(-1*self.err_x, 1*self.err_x)
        
        self.pos_y = self.pos_y + self.dt * self.vel_y
        meas_pos_y = self.pos_y + np.random.uniform(-1*self.err_y, 1*self.err_y)

        return (self.pos_x, self.pos_y), (meas_pos_x, meas_pos_y)


def nonlin_F(x, dt):
    """
    Function applies non-linear operation to state x to predict next state

    x: nx4 (x, x', y, y')
    dt: time step in s
    """

    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    return np.dot(F, x)

def nonlin_H(x):
    """
    Function simulates non-linear measurement function that transforms prior state estimate to measurement
    We measure only x,y positions

    x: nx4 (x, x', y, y'): state
    """

    # NOTE: it doesn't have to be matrix form
    return x[[0, 2]]  # returns x and y positions


if __name__ == "__main__":

    # set up simulator
    car_sim = CarSimLin2D(init_pos_xy=(10, 10), init_vel_xy=(5, 1), del_t_s=0.1, max_err_pos_x_y=(2, 2))
    
    # run simulator for 100 time steps
    gt_pos_x = []
    gt_pos_y = []
    meas_pos_x = []
    meas_pos_y = []
    for t in range(100):
        pos_gt, pos_meas = car_sim.get_sim_data()
        
        gt_pos_x.append(pos_gt[0])
        gt_pos_y.append(pos_gt[1])
        meas_pos_x.append(pos_meas[0])
        meas_pos_y.append(pos_meas[1])
    
    plt.plot(gt_pos_x, gt_pos_y, "g")
    plt.plot(meas_pos_x, meas_pos_y, "ro")
    # plt.show()

    # setup sample points generator
    sigmas = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2., kappa=1.)

    # instantiate a ukf with sigma points
    ukf = UKF(dim_x=4, dim_z=2, dt=0.1, hx=nonlin_H, fx=nonlin_F, points=sigmas)
    
    # setup other parameters for the filter
    ukf.x = np.array([5, 1, 5, 1])  # initial state
    ukf.P = np.array([[100, 0, 0, 0],
                      [0, 49, 0, 0],
                      [0, 0, 100, 0],
                      [0, 0, 0, 49]])
    # ukf.R = np.array([[9, 0, 0, 0],
    #                   [0, 1, 0, 0],
    #                   [0, 0, 9, 0],
    #                   [0, 0, 0, 1]])  # measurement cov.
    ukf.R = np.array([[16, 0],
                      [0, 16]])  # measurement cov.
    ukf.Q = np.array([[0.001, 0, 0, 0],
                      [0, 0.0001, 0, 0],
                      [0, 0, 0.001, 0],
                      [0, 0, 0, 0.0001]])
    
    # loop over simulation data and filter
    filtered_x = []
    filtered_y = []
    for index, (gt_x, gt_y, meas_x, meas_y) in enumerate(zip(gt_pos_x, gt_pos_y, meas_pos_x, meas_pos_y)):
        
        z = np.array([meas_x, meas_y])
        ukf.predict()
        ukf.update(z=z)

        filtered_x.append(ukf.x.copy()[0])
        filtered_y.append(ukf.x.copy()[2])
    
    plt.plot(filtered_x, filtered_y, "b--")
    plt.show()

    print("Done!!")
    print("Covariance UKF -> ", ukf.P)




        