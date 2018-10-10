import cv2
import numpy as np
from itertools import product
import argparse

# code for getting video as an argument by the user
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# for storing old gray image to be compared with the new
img_gray_old = 0
step = 20
angle_of_perspective = 0

def oFlowLk(oldImage, newImage, window=1):
    """
    This function calculates the flow matrix by using the formulae
    taught in the class
    """


    # initializing matrices for derivatives in x and y
    Ix = np.zeros(oldImage.shape)
    Iy = Ix.copy()

    # initializing matrix for time derivative
    It = Ix.copy()

    # calculating the derivatives in x and y
    Ix[1:-1, 1:-1] = (oldImage[1:-1, 2:] - oldImage[1:-1, :-2])/2
    Iy[1:-1, 1:-1] = (oldImage[2:, 1:-1] - oldImage[:-2, 1:-1])/2

    # calculating the time derivative
    It[1:-1, 1:-1] = oldImage[1:-1, 1:-1] - newImage[1:-1, 1:-1]

    # calculating elements of matrix 'A'
    Ix2,Iy2,Ixy = Ix * Ix,Iy * Iy,Ix * Iy

    # calculating the elements of matrix 'B'
    Ixt,Iyt = Ix * It,Iy * It

    combined = np.zeros((oldImage.shape[0], oldImage.shape[1], 5))
    combined[:,:,0], combined[:,:,1], combined[:,:,2], combined[:,:,3], combined[:,:,4] = Ix2, Iy2, Ixy, Ixt, Iyt

    # compute integration
    integrate = np.cumsum(combined, axis=0)
    integrate = np.cumsum(integrate, axis=1)

    integrate_win = (integrate[2 * window + 1:, 2 * window + 1:] - integrate[2 * window + 1:, :-1 - 2 * window] - integrate[:-1 - 2 * window, 2 * window + 1:] + integrate[:-1 - 2 * window, :-1 - 2 * window])

    # initialize flow vectors
    u = np.zeros(oldImage.shape)
    v = np.zeros(oldImage.shape)

    # for x and y derivatives
    Ix2,Iy2,Ixy =  integrate_win[:, :, 0],integrate_win[:, :, 1],integrate_win[:, :, 2]

    # for time derivative
    Ixt,Iyt = -integrate_win[:, :, 3],-integrate_win[:, :, 4]
    oFlow_matrix_x,oFlow_matrix_y = np.where(((Ix2 * Iy2) - (Ixy)**2) != 0, ((Iy2 * (-Ixt)) + ((-Ixy) * (-Iyt)))/((Ix2 * Iy2) - (Ixy)**2), 0),np.where(((Ix2 * Iy2) - (Ixy)**2) != 0, ((Ix2*(-Iyt)) + (-Ixy) * (-Ixt))/((Ix2 * Iy2) - (Ixy)**2), 0)

    # final flow matrix
    oFlow_matrix = np.zeros((oldImage.shape[0], oldImage.shape[1], 2))
    oFlow_matrix[window + 1: -1 - window, window + 1: -1 - window, 0],oFlow_matrix[window + 1: -1 - window, window + 1: -1 - window, 1] = oFlow_matrix_x[:-1, :-1],oFlow_matrix_y[:-1, :-1]
    return oFlow_matrix

def evaluateFrame(input_frame):
    """
    This function first resizes the frame, converts the frame to HSV color space
    and then apply the mask to detect a specific colored ball. Then it is converted
    to the Grayscale for final processing. Then Lucas Kanade algorithm is applied
    for calculation of the optical flow matrix. Finally the vector lines are plotted.
    """
    # color = (0,0,255) #red
    color = (255,0,0) #blue
    # color = (0,255,0) #green

    global img_gray_old

    # declare new frame that is resized from the captured one
    input_frame_new = cv2.resize(input_frame, (640, 480))

    # define values for range of the color green
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)

    # convert frame from BGR to HSV color space
    img_hsv = cv2.cvtColor(input_frame_new, cv2.COLOR_BGR2HSV)

    # create mask for the color range specified
    mask = cv2.inRange(img_hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # apply the mask to the image
    masked_img = cv2.bitwise_and(img_hsv, img_hsv, mask = mask)

    # convert HSV back to BGR
    img_bgr = cv2.cvtColor(masked_img, cv2.COLOR_HSV2BGR)

    # now convert BGR to Gray
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    sum_x,sum_y = 0,0

    velocity_x,velocity_y = 0,0

    # initialize the flow matrix
    oFlow_matrix = 0

    #check type to assert whether the old gray image array is empty or not
    if type(img_gray_old) == type(img_gray):

        oFlow_matrix = cv2.calcOpticalFlowFarneback(img_gray_old, img_gray,img_gray_old, pyr_scale=0.5, levels=4, winsize=11, iterations=8, poly_n=5, poly_sigma=1.1, flags=0)
        #oFlow_matrix = oFlowLk(img_gray_old,img_gray, window=1)

        rows = oFlow_matrix.shape[0]
        columns = oFlow_matrix.shape[1]

        pixels = product(range(0, rows, step), range(0, columns, step))

        for (i, j) in pixels:
            I_x, I_y = oFlow_matrix[i, j]
            sum_x += I_x
            sum_y += I_y
            # vector flow plotting
            cv2.line(input_frame_new, (j,i), (int(j + I_x),int(i + I_y)), color)


    img_gray_old = img_gray

    # cv2.imshow('Ball Tracker', masked_img) #result after masking the image

    # cv2.imshow('Ball Tracker', img_gray) #result for gray image after masking

    cv2.imshow('Ball Tracker', input_frame_new)

    if cv2.waitKey(1) & 0x000000FF== 27: # ESC
        return None

    return  velocity_x, velocity_y, input_frame_new


if __name__=="__main__":

    # if no video is provided in compilation command use the built-in camera
    if not args.get("video", False):

        # '0' is for built-in camera
    	cap = cv2.VideoCapture(0)

    else:

        # taking the video defined by the user
    	cap = cv2.VideoCapture(args["video"])


    while True:

        # read the captured frame
        test, input_frame = cap.read()

        if not test:
            break

        # call the main function that evaluates optical flow and plots the vector lines
        result = evaluateFrame(input_frame)

        if not result:
            break