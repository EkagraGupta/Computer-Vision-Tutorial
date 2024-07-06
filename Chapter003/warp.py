import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from homography import Haffine_from_points
from scipy import ndimage
from scipy.spatial import Delaunay


def image_in_image(im1, im2, tp):
    """Put im1 in im2 with an affine transformation such that corners
    are as close to tp as possible. tp are homogenous and counter-clockwise from top left.

    Args:
        im1 (_type_): _description_
        im2 (_type_): _description_
        tp (_type_): _description_
    """

    # points to warp from
    m, n = im1.shape[:2]
    fp = np.array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])

    print(f"Fp: {fp}\n")

    # compute affine transform and apply
    H = Haffine_from_points(fp=tp, tp=fp)
    im1_t = ndimage.affine_transform(im1, H[:2, :2], (H[0, 2], H[1, 2]), im2.shape[:2])
    print(f"Warp Im1: {im1_t}\n")
    alpha = im1_t > 0
    print(f"Alpha: {alpha}\n")
    return (1 - alpha) * im2 + alpha * im1_t


def alpha_for_triangle(points, m, n):
    """Creates alpha map of size (m, n) for a triangle with corners
    defined by points (given in normalized homogenous coordinates).

    Args:
        points (_type_): _description_
        m (_type_): _description_
        n (_type_): _description_
    """
    points = points.astype(int)
    alpha = np.zeros((m, n))
    for i in range(min(points[0]), max(points[0])):
        for j in range(min(points[1]), max(points[1])):
            x = np.linalg.solve(points, [i, j, 1])
            if min(x) > 0:  # all coefficients positive
                alpha[i, j] = 1
    return alpha


def triangulate_points(x, y):
    """Delaunay triangulation of 2D points.

    Args:
        x (_type_): _description_
        y (_type_): _description_
    """

    tri = Delaunay(np.c_[x, y]).simplices
    return tri


def pw_affine(from_im, to_im, fp, tp, tri):
    """Warp triangular patches from an image.
    from_im = image to warp
    to_im = destination image
    fp = from points in hom. coordinates
    tp = to points in hom. coordinates
    tri = triangulation.

    Args:
        from_im (_type_): _description_
        to_im (_type_): _description_
        fp (_type_): _description_
        tp (_type_): _description_
        tri (_type_): _description_
    """
    im = to_im.copy()

    # check if image is grayscale or color
    is_color = len(from_im.shape) == 3

    # create image to warp to (needed if iterate colors)
    im_t = np.zeros(im.shape, "uint8")

    for t in tri:
        # compute affine transformation
        H = Haffine_from_points(tp[:, t], fp[:, t])

        if is_color:
            for col in range(from_im.shape[2]):
                im_t[:, :, col] = ndimage.affine_transform(
                    from_im[:, :, col], H[:2, :2], (H[0, 2], H[1, 2]), im.shape[:2]
                )
        else:
            im_t = ndimage.affine_transform(
                input=from_im,
                matrix=H[:2, :2],
                offset=(H[0, 2], H[1, 2]),
                output=im.shape[:2],
            )

        # alpha for triangle
        alpha = alpha_for_triangle(points=tp[:, t], m=im.shape[0], n=im.shape[1])

        # add triangle to image
        im[alpha > 0] = im_t[alpha > 0]
    return im


def plot_mesh(x, y, tri):
    """Plot triangles."""
    for t in tri:
        t_ext = [t[0], t[1], t[2], t[0]]  # add first point to end
        plt.plot(x[t_ext], y[t_ext], "r")


if __name__ == "__main__":
    im2_path = (
        "/home/ekagra/personal/projects/ComputerVision/data/Lenna_(test_image).jpg"
    )
    im1_path = (
        "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
    )
    im1 = np.array(Image.open(im1_path).convert("L"))
    im2 = np.array(Image.open(im2_path).convert("L"))

    print(im1.shape, im2.shape)
    # set to points
    tp = np.array([[200, 500, 500, 260], [40, 36, 500, 500], [1, 1, 1, 1]])

    im3 = image_in_image(im1, im2, tp)

    plt.figure()
    plt.gray()
    plt.imshow(im3)
    plt.axis("equal")
    plt.axis("off")
    plt.show()
