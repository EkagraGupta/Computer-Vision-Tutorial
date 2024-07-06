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


def get_descriptors(image, filtered_coords, wid=5):
    """For each point return pixel values around the point using a neighborhood of width 2*wid+1.
    Assume points are extracted with min_distance>wid.

    Args:
        image (_type_): _description_
        filtered_coords (_type_): _description_
        wid (int, optional): _description_. Defaults to 5.
    """

    desc = []
    for coords in filtered_coords:
        patch = image[
            coords[0] - wid : coords[0] + wid + 1, coords[1] - wid : coords[1] + wid + 1
        ].flatten()
        desc.append(patch)
    return desc


def match(desc1, desc2, threshold=0.5):
    """For each corner point descriptor in first image,
    select its match to second image using normalized cross correlation.

    Args:
        desc1 (_type_): _description_
        desc2 (_type_): _description_
        threshold (float, optional): _description_. Defaults to 0.5.
    """

    n = len(desc1[0])
    # pair-wise distances
    d = -np.ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])

            ncc_value = np.sum(d1 * d2) / (n - 1)
            if ncc_value > threshold:
                d[i, j] = ncc_value

    n_dx = np.argsort(-d)
    match_scores = n_dx[:, 0]
    return match_scores


def match_twosided(desc1, desc2, threshold=0.5):
    """Two-sided symmetric version of match().

    Args:
        desc1 (_type_): _description_
        desc2 (_type_): _description_
        threshold (float, optional): _description_. Defaults to 0.5.
    """
    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)

    n_dx_12 = np.where(matches_12 >= 0)[0]

    # remove matches that are not symmetric
    for n in n_dx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1
    return matches_12


def append_images(im1, im2):
    """Return a new image that appends the two images side-by-side.

    Args:
        im1 (_type_): _description_
        im2 (_type_): _description_
    """

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2 - rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1 - rows2, im2.shape[1]))), axis=0)

    # if none of these cases they are equal, no filling needed
    return np.concatenate((im1, im2), axis=1)


def plot_matches(im1, im2, locs1, locs2, match_scores, show_below=True):
    """Show a figure with lines joining the accepted matches.

    Args:
        im1 (_type_): _description_
        im2 (_type_): _description_
        locs1 (_type_): _description_
        locs2 (_type_): _description_
        match_scores (_type_): _description_
        show_below (bool, optional): _description_. Defaults to True.
    """

    im3 = append_images(im1, im2)
    if show_below:
        im3 = np.vstack((im3, im3))

    plt.imshow(im3)

    cols1 = im1.shape[1]
    for i, m in enumerate(match_scores):
        if m > 0:
            plt.plot(
                [locs1[i][1], locs2[m][1] + cols1], [locs1[i][0], locs2[m][0]], "c"
            )
    plt.axis("off")


if __name__ == "__main__":
    # im_path = "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
    # im = np.array(Image.open(im_path).convert("L"))
    # harris_im = compute_harris_response(im)
    # filtered_coords = get_harris_points(harris_im)
    # plot_harris_points(im, filtered_coords)

    im1_path = (
        "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
    )
    im2_path = "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image_blurred.jpg"
    im1 = np.array(Image.open(im1_path).convert("L"))
    im2 = np.array(Image.open(im2_path).convert("L"))
    wid = 5
    harris_im = compute_harris_response(im1, 5)
    filtered_coords1 = get_harris_points(harris_im, wid + 1)
    d1 = get_descriptors(im1, filtered_coords1, wid)

    harris_im = compute_harris_response(im2, 5)
    filtered_coords2 = get_harris_points(harris_im, wid + 1)
    d2 = get_descriptors(im2, filtered_coords2, wid)

    print("starting matching...")
    matches = match_twosided(d1, d2)

    plt.figure()
    plt.gray()
    plot_matches(im1, im2, filtered_coords1, filtered_coords2, matches)
    plt.show()
