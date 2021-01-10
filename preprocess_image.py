# -*- coding: utf-8 -*

""" LAST MODIFIED: Thursday August 8th, 2019 """

"""
Development of a Machine Vision Based Yield Monitor for Shallot Onions and Carrot crops
Precision Agriculture and Sensor Systems (PASS) Research Group
McGill University, Department of Bioresource Engineering

preprocess_image.py --- This is a sub file from the yield monitoring program for the masters thesis of
Amanda Boatswain Jacques. This file preprocesses the given images using color thresholding and watershed
segmentation. It then draws circles around the detected vegetables classifying them by size,
and then retrieves the counts for all size categories.
"""

# import the necessary python libraries
import numpy as np
import cv2
import math
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage


""" Define variables and Functions """

# creates an elliptical structuring element for the opening/closing operations
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))

# define range of Red onion color in HSV
Lupper_red = np.array([50, 255, 255])
Llower_red = np.array([0,40,0])

Uupper_red = np.array([180,255, 255])
Ulower_red = np.array([160,40,0])

lower_white = np.array([0, 0, 240], dtype = "uint8")
upper_white = np.array([60, 30, 255], dtype ="uint8")

# helper functions

def ellipse_perimeter(major_axis, minor_axis):
    a = major_axis/2
    b = minor_axis/2
    h = math.pow((a-b), 2)/math.pow((a+b), 2)
    perimeter = math.pi*(a+b)*(1+ 3*h/(10 + math.sqrt(4-3*h)))
    return perimeter

def auto_canny(image, sigma=0.60):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def resize_image(image, pixel_width = 640):

    # resize the image
    r = pixel_width/image.shape[1]
    new_dim = (pixel_width, int(image.shape[0]*r))
    image = cv2.resize(image, new_dim, interpolation = cv2.INTER_AREA)

    return image

def preprocess_watershed(image, pixel_metric):
    # create a copy of the image to draw on
    clone = image.copy()
    #cv2.imshow("Original", clone)
    wts_result = np.zeros(image.shape, dtype=np.uint8)

    shifted = clone.copy()
    # convert the image to HSV colorspace and blur
    blur = cv2.medianBlur(shifted, 9)
    blur = cv2.GaussianBlur(blur, (9,9),0)
    #cv2.imshow("Blurred Image", blur)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # threshold the HSV image to get only red colors
    color_mask_lower = cv2.inRange(hsv, Llower_red, Lupper_red)
    color_mask_upper = cv2.inRange(hsv, Ulower_red, Uupper_red)
    color_mask_white = cv2.inRange(hsv, lower_white, upper_white)
    color_mask = color_mask_lower + color_mask_upper + color_mask_white
    #cv2.imshow("Color Mask", color_mask)
    color_opening = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, ellipse_kernel)
    #cv2.imshow("Opening", color_opening)
    color_closing = cv2.morphologyEx(color_opening, cv2.MORPH_CLOSE, ellipse_kernel)
    #cv2.imshow(" Closing", color_closing)

    # sure background area
    sure_bg = color_closing.copy()

    # Finding sure foreground area
    dist_transform =  ndimage.distance_transform_edt(sure_bg)
    #cv2.imshow("Distance Transform CV2", dist_transform/np.max(dist_transform[:,:]))
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
    total_extra_small = int()
    total_small = int()
    total_medium = int()
    total_large = int()
    total_extra_large = int()

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
            #cv2.imshow("Watershed Mask", mask)
            #cv2.waitKey(300)
            extra_small, small, medium, large, extra_large, wts_result = find_onion_contours(mask, clone, pixel_metric)

            clone = wts_result.copy()

            total_extra_small = total_extra_small + extra_small
            total_small = total_small + small
            total_medium = total_medium + medium
            total_large = total_large + large
            total_extra_large = total_extra_large + extra_large

    if not wts_result.any():
        wts_result = image.copy()

    return (total_extra_small, total_small, total_medium, total_large, total_extra_large), wts_result

def find_onion_contours(mask, original, pixel_metric = 1.0):
    # initialize lists of all the onion types present in the image
    extra_small_onions = []
    small_onions = []
    medium_onions = []
    large_onions = []
    extra_large_onions = []
    result = original.copy()

    # find the contours in the image
    try:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            moments = cv2.moments(c)
            #print(len(c))

            if moments['m00'] != 0.0:
                cx = int(moments['m10']/moments['m00'])
                cy = int(moments['m01']/moments['m00'])
                centroid = (int(cx), int(cy))

                if len(c) > 5:
                    # check to se if the contour is long enough to be approximated
                    # by a circle
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    diameter = radius*2

                    # filter the shapes by circularity
                    perimeter = cv2.arcLength(c, True)
                    area = cv2.contourArea(c)
                    circularity = 4*np.pi*(area/(perimeter*perimeter))
                    ratio = area/(np.pi*(radius*radius))
                    #print("ratio: ", ratio)
                    #print("circularity :", circularity)
                    #print(width)

                    width = diameter/pixel_metric

                    #if (ratio > 0.40) and (circularity > 0.60):

                    # filter the onions by size
                    if (24 <=  width < 44.45): # extra small onion
                        color = (255, 0, 0) # Blue
                        extra_small_onions.append(c)

                    if (44.45 <= width <= 46.0375): # small onion
                        small_onions.append(c)
                        color = (255, 255, 0) # Light Blue

                    if (46.0375 < width <= 47.625): # medium Onion
                        medium_onions.append(c)
                        color = (0, 255, 0) # Green

                    if (47.625 < width <= 50.80):
                        large_onions.append(c)
                        color = (0, 128, 255) # Orange

                    if (width > 50.80): # large Onion
                        extra_large_onions.append(c)
                        color = (0, 0, 255) # Red

                    if any((extra_small_onions, small_onions, medium_onions, large_onions, extra_large_onions)):
                        if  24 <= width:
                            # draw the centroid of the circle as well as the onion border
                            cv2.circle(result, centroid, 3, (255, 255, 0), -1)
                            cv2.circle(result, (int(x), int(y)), int(diameter), color, 2)

    except Exception as e:
        # if there are no contours, do nothing
        print(e)
        pass

    # return the counts for each onion category
    return len(extra_small_onions), len(small_onions), len(medium_onions), len(large_onions), len(extra_large_onions), result
