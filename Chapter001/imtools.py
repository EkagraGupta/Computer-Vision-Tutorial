import os
import numpy as np
from PIL import Image
import pylab as pl


def get_imlist(path: str) -> list:
    """Returns a list of filenames for all jpg images in a directory.

    Args:
        path (str): _description_

    Returns:
        list: _description_
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")]


def imresize(im: np.array, sz: tuple) -> Image.Image:
    """Resizes an image array using PIL.

    Args:
        im (np.array): _description_
        sz (tuple): _description_

    Returns:
        Image.Image: _description_
    """
    pil_im = Image.fromarray(im)
    return np.array(pil_im.resize(sz))


def histeq(im: np.array, num_bins: int = 256) -> any:
    """Histogram equalization of a grayscale image.
    The contrast increases and the details of the dark region enhances.

    Args:
        im (np.array): _description_
        num_bins (int, optional): _description_. Defaults to 256.

    Returns:
        any: _description_
    """
    # Get image histogram
    imhist, bins = np.histogram(im.flatten(), num_bins, density=True)
    cdf = imhist.cumsum()  # cdf
    cdf = 255 * cdf / cdf[-1]  # normalize

    # use linear interpolation of cdf to find new pixel values
    im2 = pl.interp(im.flatten(), bins[:-1], cdf)

    return im2.reshape(im.shape), cdf


def compute_average(imlist: list) -> np.array:
    """Compute the average of a list of images.
    Simple way of reducing image noise and/or artistic effects.

    Args:
        imlist (list): _description_

    Returns:
        np.array: _description_
    """
    # Open the first image and make into array of dtype float
    average_im = np.array(Image.open(imlist[0]), "f")

    for im_name in imlist[1:]:
        try:
            average_im += np.array(Image.open(im_name))
        except:
            print(im_name + "...skipped")
    average_im /= len(imlist)

    # Return average as uint8
    return np.array(average_im, "uint8")


if __name__ == "__main__":
    im_path = (
        "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpeg"
    )
    im = np.array(Image.open(im_path).convert("L"))

    im2, cdf = histeq(im=im, num_bins=256)
    pil_im = Image.fromarray(np.uint8(im))
    pil_im.show()
    pl.figure()
    pl.hist(im2.flatten(), 256)
    pl.show()
