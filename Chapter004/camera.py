import numpy as np
from scipy import linalg


class Camera(object):
    """class for representing pin-hole cameras."""

    def __init__(self, P):
        """Initialize P = K[R|t] camera model.

        Args:
            P (_type_): Projection matrix
        """
        self.P = P
        self.K = None  # calibration matrix
        self.R = None  # rotation
        self.t = None  # translation
        self.c = None  # camera center

    def project(self, X):
        """Project points in X (4*n array) and normalize coordinates."""
        x = np.dot(self.P, X)

        for i in range(3):
            x[i] /= x[2]

        return x

    def factor(self):
        """Factorize the camera matrix into K, t and R as P=K[R|t]."""

        # factor first 3*3 part
        K, R = linalg.rq(self.P[:, :3])
        
        # make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if np.linalg.det(T) < 0:
            T[1, 1] *= -1

        self.K = np.dot(K, T)
        self.R = np.dot(T, R)  # T is its own inverse
        self.t = np.dot(np.linalg.inv(self.K), self.P[:, 3])
        print(f"K: {self.K}\nR: {self.R}\nt: {self.t}")


        return self.K, self.R, self.t


def rotation_matrix(a):
    """creates a 3D rotation matrix for rotation around the axis of vector a."""

    R = np.eye(4)
    R[:3, :3] = linalg.expm([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

    return R
