from PIL import Image
import numpy as np
from scipy.ndimage import label, binary_opening

# load image and threshold to make sure its binary
im_path = "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
im = np.array(Image.open(im_path).convert("L"))
im = 1 * (im < 128)

# opening to separate objects better
im_open = binary_opening(im, np.ones((9, 5)), iterations=2)

labels, num_objects = label(im_open)
print(f"Number of objects: {num_objects }")

pil_im = Image.fromarray(im_open.astype(np.uint8))
pil_im.show()
