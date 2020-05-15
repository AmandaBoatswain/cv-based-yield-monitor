# -*- coding: utf-8 -*

""" LAST MODIFIED: Monday August 5th, 2019 """ 

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

# original preprocessing method
def preprocess_original(image, pixel_metric):
    
    # show the original image
    #cv2.imshow("Original Image", image)
    
    # perform Mean Shift Filtering
    shifted = cv2.pyrMeanShiftFiltering(image, 14, 50)
    
    # convert the image to HSV colorspace and blur 
    blur = cv2.medianBlur(shifted, 9)
    blur = cv2.GaussianBlur(blur, (9,9),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    """
    ## perform mean subtraction normalization
    
    hue_mean = np.ones(h.shape, dtype=np.uint8)*np.mean(h)
    hue_mean = hue_mean.astype(np.uint8)
    subtracted_hue = cv2.subtract(h, hue_mean)
    
    sat_mean = np.ones(s.shape, dtype=np.uint8)*np.mean(s)
    sat_mean = sat_mean.astype(np.uint8)
    subtracted_sat = cv2.subtract(s, sat_mean)
    
    val_mean = np.ones(v.shape, dtype=np.uint8)*np.mean(v)
    val_mean = val_mean.astype(np.uint8)
    subtracted_val = cv2.subtract(v, val_mean)
    subtracted = cv2.merge((subtracted_hue, subtracted_sat, subtracted_val))
    
    cv2.imshow("Mean Substracted Image", subtracted)
    cv2.imshow("Mean Subtracted Hue Channel", subtracted_hue)
    cv2.imshow("Mean Subtracted Saturation Channel", subtracted_sat)
    cv2.imshow("Mean Subtracted Value Channel", subtracted_val)
    
    #subt_ret,subt_thresh = cv2.threshold(subtracted_hue,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    """
    # threshold the HSV image to get only red colors
    color_mask_lower = cv2.inRange(hsv, Llower_red, Lupper_red)
    color_mask_upper = cv2.inRange(hsv, Ulower_red, Uupper_red)
    color_mask_white = cv2.inRange(hsv, lower_white, upper_white)
    color_mask = color_mask_lower + color_mask_upper +color_mask_white
    color_opening = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, ellipse_kernel)
    color_closing = cv2.morphologyEx(color_opening, cv2.MORPH_CLOSE, ellipse_kernel)
    #cv2.imshow("Color Segmentation", color_closing)
    
    ## chromacity calculations
    # split the BGR channels 
    b, g, r = cv2.split(blur)
    b = b.astype('float')
    g = g.astype('float')
    r = r.astype('float')
    
    # add the 3 channels together
    merged = np.add(r, b)
    merged = np.add(merged, g)
    
    # calculate red intensity
    R_I = 3*r - b -g 
    R_I = R_I.astype('uint8')
    #cv2.imshow("Red Intensity Image", R_I)
 
    # calculate chromacity of the red channel 
    chro_r = np.divide(r, merged)*255
    chro_r = chro_r.astype('uint8')
    #cv2.imshow("Chromacity Image", chro_r)
    
    # equalize red chromacity and apply a median blur, remove noise
    equ = cv2.equalizeHist(chro_r)
    equ = cv2.medianBlur(equ, 9)
    
    grayscaled = R_I.copy()
    grayscaled = cv2.equalizeHist(grayscaled)
    grayscaled= cv2.GaussianBlur(grayscaled, (9,9),0)
    
    chro_edged = cv2.Canny(grayscaled, 180, 255)
    #chro_edged = cv2.dilate(chro_edged, ellipse_kernel, iterations=1)
    #cv2.imshow("Chromacity Edged", chro_edged)
    final_mask = cv2.bitwise_and(chro_edged, color_closing)
    #cv2.imshow("Final Mask", final_mask)
   
    small, medium, large, result = find_onion_contours(final_mask.copy(), image.copy(), pixel_metric)  
    
    return (small, medium, large), result                                                       

def preprocess_watershed(image, pixel_metric):
    # create a copy of the image to draw on
    clone = image.copy() 
    #cv2.imshow("Original", clone)
    wts_result = np.zeros(image.shape, dtype=np.uint8)

    shifted = clone.copy()
    # convert the image to HSV colorspace and blur 
    blur = cv2.medianBlur(shifted, 9)
    blur = cv2.GaussianBlur(blur, (9,9),0)
    cv2.imshow("Blurred Image", blur)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # threshold the HSV image to get only red colors
    color_mask_lower = cv2.inRange(hsv, Llower_red, Lupper_red)
    color_mask_upper = cv2.inRange(hsv, Ulower_red, Uupper_red)
    color_mask_white = cv2.inRange(hsv, lower_white, upper_white)
    color_mask = color_mask_lower + color_mask_upper + color_mask_white
    cv2.imshow("Color Mask", color_mask)
    color_opening = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, ellipse_kernel)
    cv2.imshow("Opening", color_opening)
    color_closing = cv2.morphologyEx(color_opening, cv2.MORPH_CLOSE, ellipse_kernel)
    cv2.imshow(" Closing", color_closing)
    
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
            #cv2.imshow("Watershed Mask", mask)
            #cv2.waitKey(300)
            small, medium, large, wts_result = find_onion_contours(mask, clone, pixel_metric)    
            
            clone = wts_result.copy() 
                
            total_small = total_small + small
            total_medium = total_medium + medium
            total_large = total_large + large
        
    if not wts_result.any():
        wts_result = image.copy()
    
    #cv2.imshow("Final Result", wts_result)
    #cv2.waitKey(500)
            
    return (total_small, total_medium, total_large), wts_result 
        
def find_onion_contours(mask, original, pixel_metric = 1.0):
    # initialize lists of all the onion types present in the image 
    small_onions = []
    medium_onions = []
    large_onions = []
    dst, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)              
    result = original.copy()
    
    for c in contours:

        moments = cv2.moments(c)
        if moments['m00'] != 0.0:
            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])
            centroid = (int(cx), int(cy))            
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            diameter = radius*2 
                
            if len(c) > 5:
                # check to se if the contour is long enough to be approximated 
                # by a circle 
                
                # filter the onions by size 
                width = diameter/pixel_metric 
               
                if (20 <=  width < 44): # small onion
                    color = (255, 0, 0) # Blue
                    small_onions.append(c)
                    
                if (44 <= width < 54): # medium Onion
                    medium_onions.append(c)
                    color = (0, 255, 0) # Green
                    
                if (width <= 54): # large Onion
                    large_onions.append(c)
                    color = (0, 0, 255) # Red
                    
                if any((small_onions, medium_onions, large_onions)): 
                    if  20 <= width <= 64: 
                        # draw the centroid of the circle as well as the onion border 
                        cv2.circle(result, centroid, 3, (255, 255, 0), -1)
                        cv2.circle(result, (int(x), int(y)), int(diameter), color, 2)                  
        else:
            pass
    
    # return the counts for each onion category 
    return len(small_onions), len(medium_onions), len(large_onions), result 


