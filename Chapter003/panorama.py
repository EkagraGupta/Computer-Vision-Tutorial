import sift
from PIL import Image
import numpy as np

l, d = {}, {}

im1_path = "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
im1 = np.array(Image.open(im1_path).convert("L"))

l, d = sift.sift(im1)

print(type())
