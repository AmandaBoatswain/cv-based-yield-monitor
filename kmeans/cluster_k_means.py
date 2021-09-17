# -*- coding: utf-8 -*

""" LAST MODIFIED: June 5th, 2021 - 8:23 PM

Development of a Machine Vision Based Yield Monitor for Shallot Onions and 
Carrot crops Precision Agriculture and Sensor Systems (PASS) Research Group
McGill University, Department of Bioresource Engineering

cluster_k_means.py --- This is a sub file from the yield monitoring program 
for the masters thesis of Amanda A. Boatswain Jacques. This file preprocesses 
the given images using color thresholding, k-means clustering and watershed
segmentation. It then draws circles around the detected vegetables classifying 
them by size, and then retrieves the counts for all size categories.
"""

# import the necessary python libraries
from operator import xor
import numpy as np
import time
import cv2
import math
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage

""" Define variables and Functions """

# CONSTANTS

# color boundaries

# potential onion regions
LowerLowerRed = np.array([0, 0, 180])
LowerUpperRed = np.array([10, 45, 235])
UpperLowerRed = np.array([150, 20, 205])

BrightRedLower = np.array([0, 0, 190])
BrightRedUpper = np.array([60, 70, 255])

BrightPinkLower = np.array([120, 20, 175])
BrightPinkUpper = np.array([170, 60, 255])

# for the conveyor
LowerBlue = np.array([119, 29, 45])
UpperBlue = np.array([130, 55, 75])
LowerTeal = np.array([100, 75, 45])
UpperTeal = np.array([119, 135, 60])

""" OLD VALUES
BrightRedLower = np.array([0, 0, 180])
BrightRedUpper = np.array([12, 60, 255])
BrightPinkLower = np.array([130, 20, 200])
BrightPinkUpper = np.array([170, 60, 255])
"""

# specular reflection and bright spots 
upper_white = np.array([180, 15, 255], dtype ="uint8")
lower_white = np.array([0, 0, 220], dtype = "uint8")

# Morphological operations
# creates an elliptical structuring element for the opening/closing noise removal 
# operations
ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
kernel = np.ones((3,3),np.uint8)

# criteria for KMeans clustering
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
attempts=10

# FUNCTIONS 

# helper functions
def ellipse_perimeter(major_axis, minor_axis):
    a = major_axis/2
    b = minor_axis/2
    h = math.pow((a-b), 2)/math.pow((a+b), 2)
    perimeter = math.pi*(a+b)*(1+ 3*h/(10 + math.sqrt(4-3*h)))
    return perimeter

# solve the derived quadratic equation (defined in my thesis)
def calculate_mass(diameter):
    a = 0.0222
    b = 1.1239
    c = -36.483
    mass = a*diameter*diameter + b*diameter + c
    #print("mass: ", mass)
    return mass

# canny edge detection 
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

# create a function for drawing the KMEANS_PP_CENTERS. This is solely for visualization
# and debugging purposes.
def draw_canvas(centers):
    # draw a blank canvas that can fit up to 8 squares
    canvas = np.zeros((400, 800,3), dtype = "uint8")
    centers = np.uint8(centers)

    # create a list of corners
    topLeftCorners = [(0,0), (200,0), (400,0), (600, 0), (0,200), (200,200), (400,200), (600, 200)]
    bottomRightCorners = [(200,200), (400,200), (600,200),(800, 200), (200,400), (400,400), (600,400), (800, 400)]

    # draw each color in the canvas:
    for i in range(len(centers)):
        color = (int(centers[i][0]), int(centers[i][1]), int(centers[i][2]))
        cv2.rectangle(canvas, topLeftCorners[i], bottomRightCorners[i], color,-1)
        cv2.putText(canvas, str(color), (topLeftCorners[i][0]+50,topLeftCorners[i][1]+100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return canvas

# preprocess the image using the K-Means + watershed algorithm. Pixel metric 
# (a value to determine the real size of the onion, is set to 1 for now)
def preprocess_watershed(image, pixel_metric = 1.0):
    # create a copy of the image to draw on and resize it 
    img = image.copy()
    img = resize_image(img)
    clone = img.copy()
    cv2.imshow("Original", clone)
    
    # create an empty image for storing the result
    wts_result = np.zeros(image.shape, dtype=np.uint8)
    
    # blur the image twice using a gaussian filter, and once using a median filter 
    # (noise removal)
    img = cv2.GaussianBlur(img, (3,3),0)
    img = cv2.GaussianBlur(img, (3,3),0)
    img = cv2.medianBlur(img, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow("Original Image", img)

    # Apply K-Means clustering to the image
    # From analysis it looks like there are 5-6 main color clusters
    # Flatten the image
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    ret,label,centers=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)

    color_canvas = draw_canvas(centers)
    cv2.imshow("Clustered Centers", color_canvas)

    #print("Detected centers:", center)
    res = centers[label.flatten()]
    clustered_image = res.reshape((img.shape))
    cv2.imshow("Clustered Image", clustered_image)
    
    # For each different image group, determine the dominant color
    result = np.ones(clustered_image.shape[0:2], dtype="uint8")*255 # create a blank white canvas
    valid_masks = []

    for color in centers: 
        #  The next block tells the algorithm to only choose the colors that 
        # are either in the Bright red region, or bright pink region 
        if (((BrightRedLower[0] <= color[0] <= BrightRedUpper[0])
            and (BrightRedLower[1] <= color[1] <= BrightRedUpper[1])
            and (BrightRedLower[2] <= color[2] <= BrightRedUpper[2])) or
            ((BrightPinkLower[0] <= color[0] <= BrightPinkUpper[0])
            and (BrightPinkLower[1] <= color[1] <= BrightPinkUpper[1])
            and (BrightPinkLower[2] <= color[2] <= BrightPinkUpper[2]))):
            
            # retrieve these masks (red and pink) and store them
            new_mask = np.where(clustered_image[:,:,0]==color[0], result.copy(), 0)
            # save the valid masks to the new array
            valid_masks.append(new_mask)

    #print("Number of masks found: ", len(valid_masks))
    
    # If there are no valid masks, then the whole image is set as the background
    if not valid_masks:
        # Finding sure foreground area
        thresh = cv2.bitwise_not(result.copy())

    else:
        # overlay the valid onion masks on top of each other 
        final_result = sum(valid_masks)
        
        # apply some morphological operations to clean up the mask
        eroded = cv2.erode(final_result.copy(), ellipse_kernel, iterations=2)
        cv2.imshow("Eroded Image", eroded)
        dilated = cv2.dilate(eroded, ellipse_kernel, iterations=2)
        cv2.imshow("Dilated Image", dilated)
        thresh = dilated.copy()

    cv2.imshow("Final Mask", thresh)

    dist_transform = ndimage.distance_transform_edt(thresh)
    dist = cv2.normalize(dist_transform, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    cv2.imshow("Distance Transform", dist)

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this distance map
    localMax = peak_local_max(dist_transform, indices=False, min_distance=30,
                              labels=thresh)
    
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)), output = "uint8")[0]
    labels = watershed(-dist_transform, markers, mask=thresh)
    #cv2.imshow("Labelled Regions", markers[:,:])
    cv2.waitKey(1000)

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
            mask = np.zeros(wts_result.shape, dtype="uint8")
            mask[labels == label] = 255
            
            # show each detected blob one at a time
            cv2.imshow("Watershed Mask", mask)
            cv2.waitKey(20)
            
            # detect the onion counts and sizes and draw them on the image 
            extra_small, small, medium, large, extra_large, wts_result = find_onion_contours(mask, clone, pixel_metric)
            
            # rewrite the existing image 
            clone = wts_result.copy()
            
            # adjust the counts if new onions are found 
            total_extra_small = total_extra_small + extra_small
            total_small = total_small + small
            total_medium = total_medium + medium
            total_large = total_large + large
            total_extra_large = total_extra_large + extra_large

    if not wts_result.any():
        # if there is no watershed result, just return the original image
        wts_result = clone.copy()
        
    # if you wish to know vegetable counts, uncomment the following line and comment line 260    
    # return (total_extra_small, total_small, total_medium, total_large, total_extra_large), wts_result
    
    # Otherwise, just return the thresholded image and the final detection result 
    return  thresh, wts_result

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
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                    #print("radius: ", radius)
                    #print("circularity :", circularity)
                    #print(width)

                    width = diameter/pixel_metric
                    #print("width in mm: ", width)
                    mass = calculate_mass(width)
                    #print("Estimated mass: ", mass)

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

                    if (50.80 < width <= 100): # large Onion
                        extra_large_onions.append(c)
                        color = (0, 0, 255) # Red

                    if any((extra_small_onions, small_onions, medium_onions, large_onions, extra_large_onions)):
                        if  width >= 25:
                            # draw the centroid of the circle as well as the onion border
                            cv2.circle(result, centroid, 3, (255, 255, 0), -1)
                            cv2.circle(result, (int(x), int(y)), int(radius), color, 2)
                            cv2.putText(result, format(width,".3f"), (int(x+10), int(y+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    except Exception as e:
        # if there are no contours, do nothing
        print(e)
        pass

    # return the counts for each onion category and the final image 
    return len(extra_small_onions), len(small_onions), len(medium_onions), len(large_onions), len(extra_large_onions), result
