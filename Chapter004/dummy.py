import homography, sift2
import camera
import numpy as np
from PIL import Image

# load images
im1_path = '/home/ekagra/Documents/GitHub/Computer-Vision-Tutorial/data/book_frontal.jpg'
im2_path = '/home/ekagra/Documents/GitHub/Computer-Vision-Tutorial/data/book_perspective.jpg'
im1 = np.array(Image.open(im1_path).convert('L'))
im2 = np.array(Image.open(im2_path).convert('L'))

# compute features
l0, d0 = sift2.process_image(im1_path)
l1, d1 = sift2.process_image(im2_path)

# match features and estimate homography
matches = sift2.match_twosided(desc1=d0, desc2=d1)
ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2, :2].T)
model = homography.RansacModel()
H = homography.H_from_ransac(fp, tp, model)

def cube_points(c, wid):
    """Creates a list of points for plotting a cube with plot.
    (The first 5 points are the bottom square, some sides repeated)
    """
    p = []
    # bottom
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])

    # top
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])

    # vertical sides
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])

    return np.array(p).T

# camera calibration
K = camera.my_calibration((747, 1000))

# 3D points at plane z=0 with sides of length 0.2
box = cube_points([0, 0, 0.1], 0.1)

# project bottom square in first image
TBD