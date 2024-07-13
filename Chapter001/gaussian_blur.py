from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

im_path = "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
out_path = (
    "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image_blurred.jpg"
)

im = np.array(Image.open(im_path))
im2 = gaussian_filter(input=im, sigma=1)

pil_im2 = Image.fromarray(im2)
pil_im2.save(out_path)
