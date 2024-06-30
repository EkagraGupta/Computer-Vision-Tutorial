from PIL import Image
import numpy as np

im_path = "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpeg"

im = np.array(Image.open(im_path).convert("L"))
im2 = 255 - im  # invert image
im3 = (100.0 / 255) * im + 100  # clamp to interval [100:200]
im4 = (
    255.0 * (im / 255.0) ** 2
)  # squared (qudratic function which lowers the values of darker pixels)

pil_im = Image.fromarray(np.uint8(im4))
pil_im.show()
