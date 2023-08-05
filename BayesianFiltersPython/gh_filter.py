import numpy as np
import matplotlib.pyplot as plt

class TrainSim(object):
    """
    Class implements a train position tracking simulator
    Generates GT, simulates noisy measurements and also implements a GH filter to track the train
    """

    def __init__(self, init_pos_km, vel_mps, sim_time_steps, meas_error_pos_meters=500, tracker_init_pos_km=30, h=0.90, g=0.30):
        """
        Sets up the simulator -assumes a constant velocity
        """

        self.init_pos = init_pos_km * 1000  # for GT
        self.gt_pos = self.init_pos
        self.vel_mps = vel_mps
        self.sim_time_steps = sim_time_steps
        self.meas_error_pos_m = meas_error_pos_meters
        self.tracker_init_pos = tracker_init_pos_km * 1000  # only for the filter
        self.g = g  # for meas intg.
        self.h = h  # for motion model
        self.tracked_pos = self.tracker_init_pos
        self.dx = 0.5

    def get_true_positon(self):
        """
        Returns true train postion
        """

        self.gt_pos = self.gt_pos + self.vel_mps
        return self.gt_pos
    
    def measure_pos(self):
        """
        Provides a noisy measurement of train's true position
        """

        return self.gt_pos + np.random.randn() * self.meas_error_pos_m
    
    def get_tracked_pos(self, meas_pos):
        """
        Returns the tracked/ filtered train pos using the GH filter
        """
        
        
        self.pred_pos = self.tracked_pos + self.vel_mps
        res = meas_pos - self.pred_pos
        self.pred_pos = self.pred_pos + self.g * res
        self.tracked_pos = self.pred_pos
        return self.tracked_pos

        

if __name__ == "__main__":

    train_sim = TrainSim(23, 15, 100, 300)

    gt_pos = []
    meas_pos = []
    tracked_pos = []
    for t_step in range(100):
        gt_pos.append(train_sim.get_true_positon())
        meas_pos.append(train_sim.measure_pos())
        tracked_pos.append(train_sim.get_tracked_pos(meas_pos[t_step]))
    


    plt.plot(np.array(gt_pos)/ 1000, "g.")
    plt.plot(np.array(meas_pos) / 1000, "ro")
    plt.plot(np.array(tracked_pos) / 1000, "b--")
    plt.show()
