"""
Code for unscented kalman filter parts - generating sigma points using MerweScaledSigmaPoint algo. and using inscented transform to compute
mean and cov. of sigma points after passing through non-lin function
"""

from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
import numpy as np
import matplotlib.pyplot as plt


def non_lin_func(x, y):
    """
    Applies some non-lin function to sampled sigma points
    """
    return x**2, y**2

# set some initial mean and cov.
mean = np.array([0, 0])
cov = np.array([[100., 0],
                [0., 100.]])

# 2 is the dim of the state var, 2 in this case
points = MerweScaledSigmaPoints(n=2, alpha=0.3, beta=2.0, kappa=0.1)
sigmas = points.sigma_points(x=mean, P=cov)  # each is now a 2D point (x, y)

transformed_sigmas = np.zeros((5, 2))  # to hold points after transforming using non-lin transform
for index, sigma_point in enumerate(sigmas):
    non_lin_x, non_lin_y = non_lin_func(sigma_point[0], sigma_point[1])
    transformed_sigmas[index, 0], transformed_sigmas[index, 1] = non_lin_x, non_lin_y

# use unscented trasnform to get new mean and cov. of the non-lin transformed sigma points
ukf_mean, ukf_cov = unscented_transform(sigmas=transformed_sigmas, Wm=points.Wm, Wc=points.Wc)

print(sigmas)

