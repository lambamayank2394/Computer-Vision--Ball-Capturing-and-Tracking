import numpy as np
import cv2
import math
from scipy import ndimage
from itertools import product
import imutils
from numba import jit, float64, int32, int64
import argparse

# code for getting video as an argument by the user
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())


# using numba for make the code faster
@jit("float64[:,:](float64[:,:])")
def imageSmoothning(img):
    fiveXfivekernel = np.ones((5, 5), np.float32) / 25
    after5x5Smoothning = np.copy(img)
    pixels  = product(range(2, len(img) - 2),range(2, len(img[0]) - 2))
    for i,j in pixels:
        after5x5Smoothning[i][j] = (sum(map(sum, (fiveXfivekernel * img[i - 2:i + 3, j - 2:j + 3]))))
    return (after5x5Smoothning)

@jit("UniTuple(float64[:,:],2)(float64[:,:],int64)")
def edgeDetection(img,thresholdValue):
    Ix, Iy, edge_map, orientation_map = np.zeros(img.shape), np.zeros(img.shape), np.zeros(img.shape), np.zeros(
        img.shape)
    # 3x1 and 1x3 masks
    Wdx = np.array([[-0.5], [0], [0.5]]).reshape(1, 3)
    Wdy = np.array([-0.5, 0, 0.5]).reshape(1, 3)
    pixels = product(range(1, len(img) - 1), range(1, len(img[0]) - 1))
    for i,j in pixels:
        Ix[i][j] = (Wdx * img[i, j - 1:j + 2]).sum()
        Iy[i][j] = (Wdy * img[i - 1:i + 2, j]).sum()
    edge_map = np.where(np.hypot(Ix, Iy)<thresholdValue,0,150)
    orientation_map = np.arctan2(Iy, Ix)
    return (edge_map, orientation_map)

@jit("(float64[:,:])(float64[:,:],float64[:,:],float64[:,:],int64,int64,int64)")
def hough_circle(img, edge_map, orientation_map, minimumRadius, maximumRadius, distanceBtCiircle):
    row, col = img.shape
    hspace = np.zeros((maximumRadius, row, col), dtype=int)
    pixels = product(range(0, row), range(0, col))
    for a, b in pixels:
        # Getting orientation at position r,c
        theta = orientation_map[a][b]
        if edge_map[a][b]:
            # Going from specified minimum and maximum radius rather than whole 360 to speed up
            for rad in range(minimumRadius, maximumRadius):
                x, y = round(b - rad * math.cos(theta)), round(a - rad * math.sin(theta))
                if x >= 0 and x < col and y >= 0 and y < row:
                    hspace[rad][y][x] += 1
                #  Adding opposite polarity of the edge orientation too
                theta = theta +math.pi
                x, y = round(b - rad * math.cos(theta)), round(a - rad * math.sin(theta))
                if x >= 0 and x < col and y >= 0 and y < row:
                    hspace[rad][y][x] += 1
    # Non- maxima compression
    hspace[hspace < 13] = 0
    maxParameter = ndimage.maximum_filter(hspace, size=(distanceBtCiircle, distanceBtCiircle, distanceBtCiircle))
    maxParameter = np.logical_and(hspace, maxParameter)
    # Find indices of maximums
    circle_indexes = np.where(maxParameter)
    return circle_indexes

if __name__ == "__main__":

    # if no video is provided in compilation command use the built-in camera
    if not args.get("video", False):

        # '0' is for built-in camera
        cap = cv2.VideoCapture(0)

    else:

        # taking the video defined by the user
        cap = cv2.VideoCapture(args["video"])

    while True:
        ret, img = cap.read()
        # Mask values to detect only green ball
        greenLower,greenUpper = (29, 86, 6),(64, 255, 255)
        # Resizing frams to run code faster
        img = imutils.resize(img, width=200)
        cimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(cimg, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        masked_img = cv2.bitwise_and(cimg, cimg, mask=mask)
        img_smoothed = cv2.cvtColor(masked_img, cv2.COLOR_HSV2BGR)
        img_grayscale = (0.2989 * img_smoothed[:, :, 0] + 0.5870 * img_smoothed[:, :, 1] + 0.1140 * img_smoothed[:, :, 2])
        smooth_image = imageSmoothning(img_grayscale)
        edge_map, orientation_map = edgeDetection(smooth_image,10)
        circle_indexes = hough_circle(smooth_image, edge_map, orientation_map, 10, 70, 100)
        try:
            for i in range(0, len(circle_indexes[0])):
                cv2.circle(img, (int(circle_indexes[2][i]), int(circle_indexes[1][i])), int(circle_indexes[0][i]),
                           (0, 255, 0), thickness=3)
        except TypeError:
            pass
        cv2.imshow('detected circles', img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()