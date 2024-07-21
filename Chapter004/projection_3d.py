import numpy as np
from scipy import linalg

class Camera(object):
    """class for representing pin-hole cameras.
    """

    def __init__(self, P):
        """Initialize P = K[R|t] camera model.

        Args:
            P (_type_): Projection matrix
        """
        self.P = P
        self.K = None   # calibration matrix
        self.R = None   # rotation
        self.t = None   # translation
        self.c = None   # camera center

    def project(self, X):
        """Project points in X (4*n array) and normalize coordinates.
        """
        x = np.dot(self.P, X)
        
        for i in range(3):
            x[i] /= x[2]
        
        return x