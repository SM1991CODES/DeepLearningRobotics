from filterpy.stats import mahalanobis
import numpy as np

mean = np.array([[10.0, 2.0]]).T
P = np.array([[25, 0],
              [0, 16]])
z = np.array([[32, 13]])

d = mahalanobis(x=z, mean=mean, cov=P)
print(d)