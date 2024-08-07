import homography, sift2
import camera
import numpy as np
from PIL import Image

# load images
# im1_path = '/home/ekagra/Documents/GitHub/Computer-Vision-Tutorial/data/book_frontal.jpg'
# im2_path = '/home/ekagra/Documents/GitHub/Computer-Vision-Tutorial/data/book_perspective.jpg'
im1_path = '/home/ekagra/Documents/GitHub/Computer-Vision-Tutorial/data/empire_test_image.jpg'
im2_path = '/home/ekagra/Documents/GitHub/Computer-Vision-Tutorial/data/Lenna_es.jpg'
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
cam1 = camera.Camera(np.hstack((K, np.dot(K, np.array([[0], [0], [-1]])))))

# first points are the bottom square
box_cam1 = cam1.project(homography.make_homog(box[:, :5]))

# use H to transfer points to second image
box_trans = homography.normalize(np.dot(H, box_cam1))

# compute the second camera matrix from cam1 and H
cam2 = camera.Camera(np.dot(H, cam1.P))
A = np.dot(np.linalg.inv(K), cam2.P[:, :3])
A = np.array([A[:, 0], A[:, 1], np.cross(A[:, 0], A[:, 1])]).T
cam2.P[:, :3] = np.dot(K, A)

# project with the second camera
box_cam2 = cam2.project(homography.make_homog(box))


# test: projecting point on z=0 should give the same
point = np.array([1, 1, 0, 1]).T
print(homography.normalize(np.dot(np.dot(H, cam1.P), point)))
print(cam2.project(point))