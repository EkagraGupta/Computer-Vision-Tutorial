from PIL import Image
import numpy as np
from scipy.ndimage import sobel, gaussian_filter

im_path = "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
im = np.array(Image.open(im_path).convert("L"))

# # Sobel derivative filters
# im_x = np.zeros(im.shape)
# sobel(input=im, axis=1, output=im_x)

# im_y = np.zeros(im.shape)
# sobel(im, 0, im_y)

# magnitude = np.sqrt((im_x**2) + (im_y**2))

# Gaussian derivative filters
sigma = 5

im_x = np.zeros(im.shape)
gaussian_filter(im, (sigma, sigma), (0, 1), im_x)

im_y = np.zeros(im.shape)
gaussian_filter(im, (sigma, sigma), (1, 0), im_y)


pil_im = Image.fromarray(im_y)
pil_im.show()
