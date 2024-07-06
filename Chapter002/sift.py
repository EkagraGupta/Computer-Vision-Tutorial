from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def process_image(im_name, result_name, params="--edge-thresh 10 --peak-thresh 5"):
    """Process an image and save the results in a file.

    Args:
        im_name (_type_): _description_
        result_name (_type_): _description_
        params (str, optional): _description_. Defaults to '--edge-thresh 10 --peak-thresh 5'.
    """
    pgm_path = "/home/ekagra/personal/projects/ComputerVision/data/tmp.pgm"
    if im_name[-3:] != "pgm":
        # create a pgm file as binaries need the image in grayscale .pgm format
        im = Image.open(im_name).convert("L")
        im.save(pgm_path)

        im_name = pgm_path

    cmmd = str(f"sift {im_name} --output={result_name} {params}")
    os.system(cmmd)
    print(f"Processed {im_name} to {result_name}")


def read_features_from_file(filename: str):
    """Read feature properties and return in matrix form.

    Args:
        filename (str): _description_
    """

    f = np.loadtxt(filename)
    return f[:, :4], f[:, 4:]  # feature locations, descriptors


def write_features_to_file(filename, locs, desc):
    """Save feature location and descriptor to file.

    Args:
        filename (_type_): _description_
        locs (_type_): _description_
        desc (_type_): _description_
    """

    np.savetxt(filename, np.hstack((locs, desc)))


def plot_features(im, locs, circle=False):
    """Show image with features.

    Args:
        im (_type_): image as array
        locs (_type_): (row. col, scale, orientation of each feature)
        circle (bool, optional): _description_. Defaults to False.
    """

    def draw_circle(c, r):
        t = np.arange(0, 1.01, 0.01) * 2 * np.pi
        x = r * np.cos(t) + c[0]
        y = r * np.sin(t) + c[1]
        plt.plot(x, y, "b", linewidth=2)

    plt.imshow(im)

    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plt.plot(locs[:, 0], locs[:, 1], "ob")
    plt.axis("off")
