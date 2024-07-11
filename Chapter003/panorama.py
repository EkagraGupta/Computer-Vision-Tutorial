import homography
import sift
import numpy as np

def convert_points(j):
    """Converts the matches to hom. points

    Args:
        j (_type_): _description_
    """
    ndx = np.matches[j].nonzero()[0]
    fp = homography.make_homog(l[j+1][ndx, :2].T)
    ndx2 = [int(np.matches[j][i]) for i in ndx]
    tp = homography.make_homog(l[j][ndx2, :2].T)
    return fp, tp

# estimate the homographies
model = homography.RansacModel()

fp, tp = convert_points(1)