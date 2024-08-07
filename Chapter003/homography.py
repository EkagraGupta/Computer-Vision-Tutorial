import numpy as np


def normalize(points):
    """Normalize a collection of points in homogenous coordinates
    so that the last row = 1.
    """

    for row in points:
        row /= points[-1]
    return points


def make_homog(points):
    """Convert a set of points (dim*n array) to homogenous
    coordinates.
    """

    return np.vstack((points, np.ones((1, points.shape[1]))))


def H_from_points(fp, tp):
    """Find homography H, such that fp is mapped to tp
    using the linear DLT method. Points are conditioned automatically.
    """

    if fp.shape != tp.shape:
        raise RuntimeError('\nNumber of points do not match\n')

    # condition points
    # --from points--
    m = np.mean(fp[:2], axis=1)
    max_std = max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1 / max_std, 1 / max_std, 1])
    C1[0][2] = -m[0] / max_std
    C1[1][2] = -m[1] / max_std
    fp = np.dot(C1, fp)

    # --to points--
    m = np.mean(tp[:2], axis=1)
    max_std = max(np.std(tp[:2], axis=1)) + 1e-9
    C2 = np.diag([1 / max_std, 1 / max_std, 1])
    C2[0][2] = -m[0] / max_std
    C2[1][2] = -m[1] / max_std
    tp = np.dot(C2, tp)

    # create matrix for linear method, 2 rows for each corespondence pair
    num_correspondences = fp.shape[1]
    A = np.zeros((2 * num_correspondences, 9))
    for i in range(num_correspondences):
        A[2*i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0, fp[0]
                  [i]*tp[0][i], fp[1][i]*tp[0][i], tp[0][i]]
        A[2*i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1,
                      fp[0][i]*tp[1][i], fp[1][i]*tp[1][i], tp[1][i]]

    U, S, V = np.linalg.svd(A)
    H = V[8].reshape((3, 3))

    # decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    # normalize
    H /= H[2, 2]
    return H

def Haffine_from_points(fp, tp):
    """Find H, affine transformation, such that tp is affine
    transformation of fp.
    """

    if fp.shape!=tp.shape:
        raise RuntimeError('Number of points do not match.')
    
    # condition points
    # --from points--
    m = np.mean(fp[:2], axis=1)
    max_std = max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1 / max_std, 1 / max_std, 1])
    C1[0][2] = -m[0] / max_std
    C1[1][2] = -m[1] / max_std
    fp_cond = np.dot(C1, fp)

    # --to points--
    m = np.mean(tp[:2], axis=1)
    max_std = max(np.std(tp[:2], axis=1)) + 1e-9
    C2 = np.diag([1 / max_std, 1 / max_std, 1])
    C2[0][2] = -m[0] / max_std
    C2[1][2] = -m[1] / max_std
    tp_cond = np.dot(C2, tp)

    # conditioned points have mean zero, so translation is zero
    A = np.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U, S, V = np.linalg.svd(A.T)

    # create B and C matrices as Hartley-Zisserman (p 130)
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = np.concatenate((np.dot(C, np.linalg.pinv(B)), np.zeros))
    H = np.vstack((tmp2, [0, 0, 1]))

    # deconditioning
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    return H / H[2, 2]

# Example usage
if __name__ == '__main__':
    fp = np.array([[1, 2, 3],
                   [4, 5, 6]])
    tp = np.array([[1, 2, 3],
                   [4, 5, 6]])
    fp = make_homog(fp)
    tp = make_homog(tp)
    res = Haffine_from_points(fp, tp)
    print(res)
