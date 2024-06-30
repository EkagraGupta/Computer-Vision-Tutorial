import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

im_path = "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
im = np.array(Image.open(im_path).convert("L"))
im2 = gaussian_filter(im, sigma=1)
# quotient image
im2[im2 == 0] = 1
im_new = im / im2
# Normalize the result to the range [0, 255]
im_new = 255 * (im_new - np.min(im_new)) / (np.max(im_new) - np.min(im_new))

# Convert to uint8
im_new = im_new.astype(np.uint8)

# Convert to PIL image and display
pil_im_new = Image.fromarray(im_new)
pil_im_new.show()
