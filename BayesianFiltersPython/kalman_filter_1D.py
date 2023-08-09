import numpy as np
import matplotlib.pyplot as plt

"""
Implementation of a 1D kalman filter for tracking position
Assumption - we can measure position directly, process model is linear for most part
"""

class Gaussians(object):
    """
    Class implements gaussian random numbers
    """
    def __init__(self, name, mean, std) -> None:
        """
        Default constructor - sets up the Gaussian random variable
        """

        self.name = name
        self.mean = mean
        self.std = std
        self.var = self.std ** 2
    
    def print_var(self):
        """
        Simply prints the Gaussian rv params
        """
        print("Name -> {0}, mean -> {1}, var -> {2}".format(self.name, self.mean, self.var))

def sum_gaussians(g1:Gaussians, g2:Gaussians):
    """
    Implements sum of 2 Gaussians
    """

    mu = g1.mean + g2.mean
    var = g1.var + g2.var
    std = np.sqrt(var)
    name = g1.name + '+' + g2.name
    g = Gaussians(name, mu, std)
    return g

def product_gaussians(g1:Gaussians, g2:Gaussians):
    """
    Function mutliplies two Gaussians
    """

    mu = ((g1.var * g2.mean) + (g2.var * g1.mean)) / (g1.var + g2.var)
    var = (g1.var * g2.var) / (g1.var + g2.var)
    name = g1.name + 'x' + g2.name
    g = Gaussians(name, mu, var)
    return g

def meas_function(data):
    """
    Simulates a sensor - can measure only integer changes
    """

    return data.astype(np.int32)

def get_simulation_data(n_t_steps):
    """
    Function generates position and measurement data for given number of timesteps
    """
    pos_gt = np.arange(0, n_t_steps, dtype=np.float32)  # linear position change
    pos_noise = np.random.uniform(0, 2, pos_gt.shape)
    pos_gt += pos_noise

    meas = meas_function(pos_gt).astype(np.float32)
    meas_noise = np.random.uniform(-2, 3, pos_gt.shape)
    meas += meas_noise

    return pos_gt, meas


class KalmanFilter1D(object):
    """
    Class implements the 1D Kalman filter
    """

    def __init__(self, g_init_pos:Gaussians) -> None:
        """
        Default constructor - set's up parameters foe the KF
        """

        self.x = g_init_pos
        print("Filter init done")
        g_init_pos.print_var()
    
    def predict(self, g_vel:Gaussians):
        """
        Computes the predicted next state using the process model and velocity as a Gaussian
        """
        self.x_hat = sum_gaussians(self.x, g_vel)  # since we assume unit time step
        return self.x_hat

    def update(self, g_meas:Gaussians):
        """
        Function updates the predicted next state using measurement
        """
        self.x = product_gaussians(self.x_hat, g_meas)
        print("State variance -> ", self.x.var)
        return self.x


if __name__ == "__main__":

    g1 = Gaussians("pos", 10, 5)
    g2 = Gaussians("meas", 11, 7)
    g3 = sum_gaussians(g1, g2)
    g3.print_var()

    g3 = product_gaussians(g1, g2)
    g3.print_var()

    pos_gt, meas = get_simulation_data(100)
    plt.plot(pos_gt, "g")
    plt.plot(meas, "r")
    # plt.show()

    k_filter = KalmanFilter1D(Gaussians("e_pos", 10, 2))
    pred_states = []  # from prediction function of Kalman filter
    filtered_states = [] # from update function of Kalman filter
    for index, (gt, meas) in enumerate(zip(pos_gt, meas)):

        x_hat = k_filter.predict(Gaussians("vel", 1.0, 1.0))
        pred_states.append(x_hat.mean)
        x_hat = k_filter.update(Gaussians("meas", meas, 20.0))
        filtered_states.append(x_hat.mean)
    
    plt.plot(pred_states, "c")
    plt.plot(filtered_states, "b")
    plt.show()

