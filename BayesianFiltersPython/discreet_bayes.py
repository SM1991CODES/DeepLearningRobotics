"""
Code from chapter 2 - Discreet bayes filter
"""

import numpy as np
import matplotlib.pyplot as plt

class DogTracker(object):
    """
    Class implements a discreet Baye's filter to track a dog through a hallway
    We know the map of hallway, it has 10 positions of which three are doors and others are walls
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 1], 1-> door, 0-> wall
    Dog's sensor reads door/ wall with some error

    Args:
        object (_type_): _description_
    """

    def __init__(self, hall) -> None:
        """
        Default constructor - set's the environment and assigns equal probability to each position

        Args:
            hall (numpy array): map of the hallway as described above
        """

        self.hall = hall
        n_pos = len(self.hall)
        p_init = 1 / n_pos
        self.belief = np.ones_like(self.hall) * p_init  # initial belief for each position of dog - equal for all
        print("Tracker initialized...")
    
    def update_belief(self, meas, p_correct):
        """
        Method updates the beliefs based on measurement
        Since measurements are noisy, p_correct is probability of correct measurements

        Args:
            meas (int): 1 or 0
            p_correct (float): probability of correct measurement by sensor
        """

        self.belief[self.hall == meas] *= (p_correct / (1 - p_correct))
        print("Sum of belief after update = ", self.belief.sum())

        # normalize belief to make it probability distr. again
        self.belief /= self.belief.sum()
        return self.belief


def gaussian_probability(x, mu, std):
    """
    Function returns probability of P(X = x) if x is from a Gaussian distribution with mean mu and std. dev. std

    Args:
        x (float): value for gaussian random variable X
    """

    p_x = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-1 * ((x - mu)**2) / (2 * (std**2)))
    return p_x


if __name__ == "__main__":

    tracker = DogTracker(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1]))
    bel = tracker.update_belief(1, 0.75)
    plt.bar(np.arange(0, 10), bel)
    plt.show()
    plt.close()

    x = np.arange(1, 60)
    p_x = gaussian_probability(x, 50, 0.5)
    plt.plot(x, p_x)
    plt.show()




    
