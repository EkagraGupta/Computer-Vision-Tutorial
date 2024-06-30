import numpy as np
from PIL import Image

im_path1 = "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
im_path2 = (
    "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image_blurred.jpg"
)
original_im = np.array(Image.open(im_path1).convert("RGB"))
blurred_im = np.array(Image.open(im_path2).convert("RGB"))

residual_im = blurred_im - original_im
pil_residual_im = Image.fromarray(residual_im)
pil_residual_im.show()
