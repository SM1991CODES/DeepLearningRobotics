"""
Illustrates UKF application to a more complex problem of tracking an airplane with a radar
We want to track distance along x on ground and elevation.
Radar returns range and elevation.

State: [x, x', y, y'], where x: distance from radar station on ground, x': velocity, y: altitude, y': climb rate/ rate of change of altitude

Measurement function: Should convert state X to measurement [r, ele]
r = sqrt((x_plane - x_radar)^2 + (y_plane - y_radar)^2)
ele = tanh( (y_plane - y_radar) / (x_plane - x_radar))

Uses built in filterpy functions to demonstrate usage, does not implement everything

TODO: Filter diverging - needs rework
"""

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise


class AirSim(object):
    """
    Class implements an aircraft simulation.
    """

    def __init__(self, init_pos_x, init_vel_x, init_height_y, init_height_change, time_step_s) -> None:
        """
        Default constructor - sets up the simulator

        """

        self.pos = init_pos_x
        self.vel = init_vel_x
        self.height = init_height_y
        self.climb_rate = init_height_change
        self.dt = time_step_s
        self.sim_steps = 0

    def measure_radar(self, pos, height):
        """
        Simulates mesurement with a radar to return range and elevation angle for a given distance and height
        """

        # we assume radar is at origin
        self.radar_pos = 0
        self.radar_height = 0
        r = np.sqrt( (pos - self.radar_pos)**2  + (height - self.radar_height) ** 2)
        ele = np.arctan2( (height - self.radar_height), (pos - self.radar_pos))
        
        return r, ele
    
    def verify_measurement(self, r, ele, pos, height):
        """
        Helper function to verify conversion between position-height and range-elevation angle worked ok
        """

        h = r * np.sin(ele)
        p = r * np.cos(ele)
        print("== measurement (r, theta) -> (pos, height): ", p, h)
        print("== GT pos, height : ", pos, height)

    def get_simulation_data(self):
        """
        Forwards simulation by 1 time step and returns GT and measurement data
        Simulates the plane's movement and then measures using the virtual radar, also verifies measurement
        """

        self.pos = self.pos + self.dt * self.vel
        
        if self.sim_steps > 50:
            self.height = self.height + self.dt * self.climb_rate
        else:
            self.height = 0

        # add noise to simulate noisy measurement with radar
        noise_pos = np.random.uniform(-3, 3)
        noise_height = np.random.uniform(-3, 3)

        meas_r, meas_ele = self.measure_radar(self.pos + noise_pos, self.height + noise_height)

        self.verify_measurement(r=meas_r, ele=meas_ele, pos=self.pos, height=self.height)

        self.sim_steps += 1

        return self.pos, self.height, meas_r, meas_ele
    

def f_radar(x, dt):
    """
    Function to apply state transition
    State X = [x, x', y, y']
    """

    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    return np.dot(F, x)


def h_radar(x):
    """
    Measurement function - converts state to measurement
    measurement: [range, elevation]
    """

    radar_pos = 0
    radar_height = 0
    pos, height = x[0], x[2]
    r = np.sqrt( (pos - radar_pos)**2  + (height - radar_height) ** 2)
    ele = np.arctan2( (height - radar_height), (pos - radar_pos))
    return np.array([r, ele])


if __name__ == "__main__":

    # ============================================= UKF part ==================================== #

    # create sigma points and UKF instance
    dt = 1
    points = MerweScaledSigmaPoints(n=4, alpha=.1, beta=2., kappa=1)
    kf = UKF(dim_x=4, dim_z=2, dt=1, fx=f_radar, hx=h_radar, points=points)
    
    # assign filter params
    kf.x = [0, 10, 15000, 300]
    kf.R = np.array([[25, 0],
                     [0, 4]])
    # kf.Q = np.array([[0.0001, 0, 0, 0],
    #                  [0, 0.00001, 0, 0],
    #                  [0, 0, 0.0001, 0],
    #                  [0, 0, 0, 0.00001]])
    kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.1)
    kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=0.1)
    kf.P = np.array([[200**2, 0, 0, 0],
                     [0, 9, 0, 0],
                     [0, 0, 150**2, 0],
                     [0, 0, 0, 9]])
    
    # =========================================================================================== #
    
    # instantiate and run simulator for 100 time steps
    gt_pos = []
    gt_height = []
    meas_range = []
    meas_ele = []
    filt_pos = []
    filt_height = []
    airsim = AirSim(init_pos_x=0, init_vel_x=15, init_height_y=20000, init_height_change=200, time_step_s=0.5)
    for t in range(100):
        pos_gt, height_gt, range_meas, ele_meas = airsim.get_simulation_data()
        
        gt_pos.append(pos_gt)
        gt_height.append(height_gt)
        meas_range.append(range_meas)
        meas_ele.append(ele_meas)

        kf.predict()
        kf.update(np.array([range_meas, ele_meas]))
        filt_pos.append(kf.x[0])
        filt_height.append(kf.x[2])

        print("========== i = {0}, P = {1} ==========".format(t, kf.P))

    plt.plot(gt_pos, gt_height, "g")
    plt.plot(filt_pos, filt_height, "r")
    plt.show()

    


    print("Done")

        
