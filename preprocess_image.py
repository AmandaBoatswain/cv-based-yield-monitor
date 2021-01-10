# -*- coding: utf-8 -*-
""" LAST MODIFIED: Sunday, January 10th, 2021 """

"""
preprocess_image.py

@author: Amanda

Machine Vision Yield Monitor Program

Development of a Machine Vision Based Yield Monitor for Shallot Onions and Carrot crops
Precision Agriculture and Sensor Systems (PASS) Research Group
McGill University, Department of Bioresource Engineering

preprocess_image.py --- This is a sub file from the yield monitoring program for the masters thesis of
Amanda A. Boatswain Jacques. This file preprocesses the given images using color thresholding and watershed
segmentation. It then draws circles around the detected vegetables classifying them by size,
and retrieves the counts for all size categories.
"""

""" Import Libraries """
import numpy as np
import cv2
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

""" Define variables and Functions """
# creates an elliptical structuring element for the opening/closing operations
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))

# define ranges of Red onion color in HSV
Lupper_red = np.array([40, 255, 255])
Llower_red = np.array([0,40,0])

Uupper_red = np.array([180,255, 255])
Ulower_red = np.array([160,40,0])

# define white regions to account for specular reflection
lower_white = np.array([0, 0, 240], dtype = "uint8")
upper_white = np.array([60, 30, 255], dtype ="uint8")


def preprocess_watershed(image, pixel_metric):
    # create a copy of the image to draw on
    clone = image.copy()
    wts_result = np.zeros(image.shape, dtype=np.uint8)

    shifted = clone.copy()
    # convert the image to HSV colorspace and blur
    blur = cv2.medianBlur(shifted, 9)
    blur = cv2.GaussianBlur(blur, (9,9),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # threshold the HSV image to get only red colors
    color_mask_lower = cv2.inRange(hsv, Llower_red, Lupper_red)
    color_mask_upper = cv2.inRange(hsv, Ulower_red, Uupper_red)
    color_mask_white = cv2.inRange(hsv, lower_white, upper_white)
    color_mask = color_mask_lower + color_mask_upper + color_mask_white

    # Apply morphological opening and closing using the ellipse-shaped kernel
    color_opening = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, ellipse_kernel)
    color_closing = cv2.morphologyEx(color_opening, cv2.MORPH_CLOSE, ellipse_kernel)

    """ Marker Based Watershed Segmentation """
    # sure background area
    sure_bg = color_closing.copy()

    # Finding sure foreground area
    dist_transform =  ndimage.distance_transform_edt(sure_bg)
    thresh = color_closing.copy()

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this distance map
    localMax = peak_local_max(dist_transform, indices=False, min_distance=25,
                              labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-dist_transform, markers, mask=thresh)

    # loop over the unique labels returned by the Watershed algorithm
    total_small = int()
    total_medium = int()
    total_large = int()

    if (len(np.unique(labels)) - 1) >= 1:
        for label in np.unique(labels):
            # if the label is zero, we are examining the 'background'
            # so simply ignore it
            if label == 0:
                continue
            # otherwise, allocate memory for the label region and draw
            # it on the mask
            mask = np.zeros(color_closing.shape, dtype="uint8")
            mask[labels == label] = 255
            small, medium, large, wts_result = find_onion_contours(mask, clone, pixel_metric)

            clone = wts_result.copy()

            total_small = total_small + small
            total_medium = total_medium + medium
            total_large = total_large + large

    # if no onions are detected, return the original image to display on the monitor
    if not wts_result.any():
        wts_result = image.copy()

    return (total_small, total_medium, total_large), wts_result

def find_onion_contours(mask, original, pixel_metric = 1.0):
    # initialize lists of all the onion types present in the image
    small_onions = []
    medium_onions = []
    large_onions = []
    result = original.copy()

    # find the contours in the image
    try:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            moments = cv2.moments(c)

            # find the position of the onion's centroid
            if moments['m00'] != 0.0:
                cx = int(moments['m10']/moments['m00'])
                cy = int(moments['m01']/moments['m00'])
                centroid = (int(cx), int(cy))

                if len(c) > 5:
                    # check to se if the contour is long enough to be approximated
                    # by a circle
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    diameter = radius*2

                    # calculate the width of the onion in mm
                    width = diameter/pixel_metric

                    if len(c) > 5:

                        # filter the onions by size
                        if (20 <=  width < 44):
                            # small onion
                            color = (255, 0, 0) # Blue
                            small_onions.append(c)

                        if (44 <= width < 54):
                            # medium Onion
                            medium_onions.append(c)
                            color = (0, 255, 0) # Green

                        if (54 <= width <= 80):
                            # large Onion
                            large_onions.append(c)
                            color = (0, 0, 255) # Red

                        # if any onions are detected, draw them on the image
                        if any((small_onions, medium_onions, large_onions)):
                            if  20 <= width:
                            # draw the centroid of the circle as well as the onion border
                            cv2.circle(result, centroid, 3, (255, 255, 0), -1)
                            cv2.circle(result, (int(x), int(y)), int(diameter), color, 2)

    except Exception as e:
        # if there are no contours, do nothing
        print(e)
        pass

    # return the counts for each onion category
    return len(small_onions), len(medium_onions), len(large_onions), result
