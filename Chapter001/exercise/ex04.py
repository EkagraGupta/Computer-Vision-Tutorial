import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import sobel


def find_outlines(im: Image.Image, threshold: int = 100):
    im_x = np.zeros(im.shape)
    sobel(input=im, axis=0, output=im_x)
    im_y = np.zeros(im.shape)
    sobel(input=im, axis=1, output=im_y)

    edge_mag = np.sqrt(im_x**2 + im_y**2)
    edge_mag = (edge_mag / np.max(edge_mag)) * 255.0
    edge_map = edge_mag > threshold
    pil_edge_map = Image.fromarray(edge_map)
    return pil_edge_map


if __name__ == "__main__":
    im_path = "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
    im = np.array(Image.open(im_path).convert("L"))
    # im = ImageOps.autocontrast(im)
    # im = np.array(im)
    pil_im = find_outlines(im=im)
    pil_im.show()
