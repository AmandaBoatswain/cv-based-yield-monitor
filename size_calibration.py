# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 00:19:31 2018
@author: amanda

size_calibration.py -- Size calibration file for determining onion sizes. Place the
tennis ball in the camera FOV, and record a few images. Change the calibration
directory accordingly in the conf.json file.
"""
# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import os.path
import imutils
import cv2

def calibrate(directory):
    # ================== Load Calibration Data ====================================

    # folder with  the calibration images
    calib_directory = directory

    # ================== Variables ================================================
    ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    metrics = []
    width = 65.4 # width of tennis ball in mm

    # boundaries for color segmentation
    # H: 0 - 180, S: 0 - 255, V: 0 - 255
    upper_yellow = np.array([60, 255, 255] , dtype = "uint8")
    lower_yellow = np.array([30, 50, 50], dtype = "uint8")

    lower_white = np.array([0, 0, 240], dtype = "uint8")
    upper_white = np.array([180, 60, 255], dtype ="uint8")

    # Loop through all the calibration photos
    print("[INFO] Loading Calibration images...\n")

    for root, dirs, filenames in os.walk(calib_directory):
        for i, file in enumerate(filenames):
            imgpath = os.path.join(root,file) # Reconstructs the file path using
            # the root_directory and current filename
            #print(imgpath)

            if imgpath.find(".ini") > -1:
                pass
            else:
                img = cv2.imread(imgpath)
                (h, w) = img.shape[:2]

                # resize to half-width, blur and color threshold
                #img = imutils.resize(img, int(w/2))
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                blur = cv2.GaussianBlur(hsv, (7,7), 0)

                # perform edge detection, then perform a dilation + erosion to
                # close gaps in between object edges
                mask = cv2.inRange(blur,lower_yellow, upper_yellow)
                mask2 = cv2.inRange(blur, lower_white, upper_white)
                cv2.imshow("White Mask", mask2)
                cv2.imshow("Plain Mask", mask)

                mask = cv2.bitwise_xor(mask, mask2)
                eroded = cv2.erode(mask.copy(), ellipse_kernel, iterations=3)
                dilated = cv2.dilate(eroded, ellipse_kernel, iterations=3)
                cv2.imshow("Dilated Mask", dilated)

                # Find the contour of the tennis ball
                cnts, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
                #cnts = cnts[0] if imutils.is_cv2() else cnts[1]

                # loop over the contours
                for c in cnts:
                    # fit a minimum enclosing circle to the contour
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    if radius >=100:
                        clone = img.copy() # create a copy of the original to draw on
                        # draw a circle around the ball
                        cv2.circle(clone, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                        # define and draw extremity points on the circle d_diameter
                        (left_midpoint, right_midpoint) = x-radius, x+radius
                        left_coordinate = (left_midpoint, y)
                        right_coordinate = (right_midpoint, y)
                        cv2.circle(clone, (int(left_midpoint), int(y)), 5, (0, 0, 255), -1)
                        cv2.circle(clone, (int(right_midpoint), int(y)), 5, (0, 0, 255), -1)
                        cv2.line(clone, (int(left_midpoint), int(y)),(int(right_midpoint),
                                     int(y)), (255,255,255),2)
                        # draw the center of the circle
                        cv2.circle(clone, (int(x), int(y)), 5, (0, 0, 255), -1)
                        cv2.imshow("Calibration Image", clone)
                        cv2.waitKey(500)
                        # compute the Euclidean distance between the midpoints
                        d_diameter = dist.euclidean(left_coordinate, right_coordinate)
                        metric = d_diameter/width
                        metrics.append(metric)

            PixelsPerMetric = np.mean(metrics)

        cv2.destroyAllWindows()

    return PixelsPerMetric
