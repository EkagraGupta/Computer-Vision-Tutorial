import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def compute_harris_response(im, sigma=3):
    """Compute the Harris corner detector response function
    for each pixel in the grayscale image.

    Args:
        im (_type_): _description_
        sigma (int, optional): _description_. Defaults to 3.
    """

    # derivatives
    im_x = np.zeros(im.shape)
    gaussian_filter(im, (sigma, sigma), (0, 1), im_x)
    im_y = np.zeros(im.shape)
    gaussian_filter(im, (sigma, sigma), (1, 0), im_y)

    # compute components of the harris matrix
    Wxx = gaussian_filter(im_x * im_x, sigma)
    Wxy = gaussian_filter(im_x * im_y, sigma)
    Wyy = gaussian_filter(im_y * im_y, sigma)

    # determinant and trace
    det_W = Wxx * Wyy - Wxy**2
    tr_W = Wxx + Wyy

    return det_W / tr_W


def get_harris_points(harris_im, min_dist: int = 10, threshold: float = 0.01):
    """Returns corners from a Harris response image.
    min_dist is the minimum number of pixels separating corners and image boundary.

    Args:
        harris_im (_type_): _description_
        min_dist (int, optional): _description_. Defaults to 10.
        threshold (float, optional): _description_. Defaults to 0.1.
    """

    # find top corner candidates above a threshold
    corner_threshold = harris_im.max() * threshold
    harris_im_t = (
        harris_im > corner_threshold
    ) * 1  # creates a binary image where pixels with Harris response above threshold are
    # set to 1 (corner candidates) and others 0.

    # get coordinates of candidates
    coords = np.array(harris_im_t.nonzero()).T

    # ... and their values
    candidate_values = [harris_im[c[0], c[1]] for c in coords]

    # sort candidates
    index = np.argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = np.zeros(harris_im.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # select the best points taking min_dist into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[
                (coords[i, 0] - min_dist) : (coords[i, 0] + min_dist),
                (coords[i, 1] - min_dist) : (coords[i, 1] + min_dist),
            ] = 0
    return filtered_coords


def plot_harris_points(image, filtered_coords):
    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], "o")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    im_path = "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
    im = np.array(Image.open(im_path).convert("L"))
    harris_im = compute_harris_response(im)
    filtered_coords = get_harris_points(harris_im)
    plot_harris_points(im, filtered_coords)
