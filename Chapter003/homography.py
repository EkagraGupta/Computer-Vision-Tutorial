import numpy as np


def normalize(points):
    """Normalize a collection of points in homogenous coordinates so that
    the last row = 1.

    Args:
        points (_type_): _description_
    """

    for row in points:
        row /= points[-1]
    return points


def make_homog(points):
    """Convert a set of points (dim*n array) to homogeneous coordinates.

    Args:
        points (_type_): _description_
    """

    return np.vstack((points, np.ones((1, points.shape[1]))))


def H_from_points(fp, tp):
    """Find homography H, such that fp is mapped to tp using the linear
    DLT method. Points are conditioned automatically.

    Args:
        fp (_type_): _description_
        tp (_type_): _description_
    """

    if fp.shape != tp.shape:
        raise RuntimeError("Number of points do not match.")

    # condition points (important for numerical reasons)
    # --from points--
    print("----- FROM POINTS -----")
    m = np.mean(fp[:2], axis=1)
    max_std = np.max(np.std(fp[:2], axis=1)) + 1e-9
    print(f"Mean: {m}\tMax Std: {max_std}\n")
    C1 = np.diag([1 / max_std, 1 / max_std, 1])
    print(f"Initial C1: {C1}\n")
    C1[0][2] = -m[0] / max_std
    C1[1][2] = -m[1] / max_std
    fp = np.dot(C1, fp)
    print(f"C1 after: {C1}\nFp: {fp}\n")

    # --to points--
    print("----- TO POINTS -----")
    m = np.mean(tp[:2], axis=1)
    max_std = np.max(np.std(tp[:2], axis=1)) + 1e-9
    print(f"Mean: {m}\tMax Std: {max_std}\n")
    C2 = np.diag([1 / max_std, 1 / max_std, 1])
    print(f"Initial C2: {C2}\n")
    C2[0][2] = -m[0] / max_std
    C2[1][2] = -m[1] / max_std
    tp = np.dot(C2, tp)
    print(f"C2 after: {C2}\nTp: {tp}\n")

    # Create matrix for linear method, 2 rows for each correspondence pair
    num_correspondences = fp.shape[1]
    print(f"Num Correspondences: {num_correspondences}\n")
    A = np.zeros((2 * num_correspondences, 9))
    print(f"Matr A: {A}\n")
    for i in range(num_correspondences):
        A[2 * i] = [
            -fp[0][i],
            -fp[1][i],
            -1,
            0,
            0,
            0,
            tp[0][i] * fp[0][i],
            tp[0][i] * fp[1][i],
            tp[0][i],
        ]
        A[2 * i + 1] = [
            0,
            0,
            0,
            -fp[0][i],
            -fp[1][i],
            -1,
            tp[1][i] * fp[0][i],
            tp[1][i] * fp[1][i],
            tp[1][i],
        ]
    print(f"A after: {A}\n")
    U, S, V = np.linalg.svd(A)
    print(f"SVD...\nDesired matr V: {V}\n")
    H = V[8].reshape((3, 3))
    print(f"V reshaped into 3x3 (H): {H}\n")

    # decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))
    print(f"Decondition H: {H}\n")

    # normalize and return
    print(f"Normalized H: {H / H[2, 2]}\n")
    return H / H[2, 2]


def Haffine_from_points(fp, tp):
    """Find H, affine transformation, such that tp is affine transoformation
    of fp.

    Args:
        fp (_type_): _description_
        tp (_type_): _description_
    """

    if fp.shape != tp.shape:
        raise RuntimeError("Number of points do not match")

    # Condition points
    # --from points--
    print(f"----- FROM POINTS -----")
    m = np.mean(fp[:2], axis=1)
    max_std = np.max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1 / max_std, 1 / max_std, 1])
    C1[0][2] = -m[0] / max_std
    C1[1][2] = -m[1] / max_std
    fp_cond = np.dot(C1, fp)
    print(f"C1: {C1}\nFp conditioned: {fp_cond}\n")

    # --to points--
    print(f"----- TO POINTS -----")
    m = np.mean(tp[:2], axis=1)
    C2 = C1.copy()  # must use same scaling for both point sets
    C2[0][2] = -m[0] / max_std
    C2[1][2] = -m[1] / max_std
    tp_cond = np.dot(C2, tp)
    print(f"C2: {C2}\nTp conditioned: {tp_cond}\n")

    # conditioned points have mean 0, so translation is 0
    A = np.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    print(f"Matr. A: {A}\n")
    U, S, V = np.linalg.svd(A.T)
    print(f"SVD...\nV: {V}\n")
    # create B and C matrixes as Hartley-Zisserman
    tmp = V[:2].T
    B, C = tmp[:2], tmp[2:4]
    print(f"Matr B: {B}\nMatr C: {C}\n")

    tmp2 = np.concatenate((np.dot(C, np.linalg.pinv(B)), np.zeros((2, 1))), axis=1)
    print(f"Tmp2: {tmp2}\n")
    H = np.vstack((tmp2, [0, 0, 1]))
    print(f"H: {H}\n")

    # decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))
    print(f"Deconditioned H: {H}\n")

    return H / H[2, 2]


if __name__ == "__main__":
    # Example points
    fp = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]])

    tp = np.array([[0, 2, 2, 0], [0, 0, 3, 3], [1, 1, 1, 1]])

    H = Haffine_from_points(fp, tp)
    print(f"Homography matrix: {H}")
