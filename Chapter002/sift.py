from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


def sift(im):
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(im, None)
    return kp, desc


def plot_matches(im1, im2, keypoints1, keypoints2, matches):
    im_matches = cv2.drawMatches(
        im1,
        keypoints1,
        im2,
        keypoints2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.imshow(im_matches)
    plt.show()


def match(desc1, desc2, threshold=0.75):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    return good_matches


def match_twosided(desc1, desc2, threshold=0.75):
    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)
    print(matches_12)
    matches_12_dict = {m.queryIdx: m for m in matches_12}
    matches_21_dict = {m.queryIdx: m for m in matches_21}

    # Ensure matches are symmetric
    symmetric_matches = []
    for m in matches_12:
        if (
            m.trainIdx in matches_21_dict
            and matches_21_dict[m.trainIdx].trainIdx == m.queryIdx
        ):
            symmetric_matches.append(m)

    return symmetric_matches


if __name__ == "__main__":
    im1_path = (
        "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
    )
    im2_path = "/home/ekagra/personal/projects/ComputerVision/data/Lenna_es.jpg"
    im1 = np.array(Image.open(im1_path).convert("L"))
    im2 = np.array(Image.open(im2_path).convert("L"))
    keypoints1, descriptors1 = sift(im1)
    keypoints2, descriptors2 = sift(im2)
    matches = match_twosided(desc1=descriptors1, desc2=descriptors2)
    print(matches)
    # Plot matches
    plot_matches(im1, im2, keypoints1, keypoints2, matches)
    # plt.figure()
    # plt.gray()
    # plot_matches(im1=im1,
    #              im2=im2,
    #              locs1=keypoints1,
    #              locs2=keypoints2,
    #              match_scores=matches)
    # plt.show()
