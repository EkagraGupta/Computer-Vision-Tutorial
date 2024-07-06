import homography
import warp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# open image to warp
im1_path = "/home/ekagra/personal/projects/ComputerVision/data/sunset_tree.jpg"
from_im = np.array(Image.open(im1_path))
x, y = np.meshgrid(range(5), range(6))
x = (from_im.shape[1] / 4) * x.flatten()
y = (from_im.shape[0] / 5) * y.flatten()

print(x, y)

# triangulate
tri = warp.triangulate_points(x, y)

# open image and destination points
im = np.array(
    Image.open("/home/ekagra/personal/projects/ComputerVision/data/turningtorso.jpg")
)
tp = np.loadtxt(
    "/home/ekagra/personal/projects/ComputerVision/data/turningtorso.txt"
)  # destination points

# convert points to hom.coordinates
fp = np.vstack((y, x, np.ones((1, len(x)))))
tp = np.vstack((tp[:, 1], tp[:, 0], np.ones((1, len(tp)))))

# warp triangles
im = warp.pw_affine(from_im=from_im, to_im=im, fp=fp, tp=tp, tri=tri)

# plot
plt.figure()
plt.imshow(im)
warp.plot_mesh(tp[1], tp[0], tri)
plt.axis("off")
plt.show()
