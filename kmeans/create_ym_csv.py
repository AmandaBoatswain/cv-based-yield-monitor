""" A file to recreate new data using the improved segmentation method. This file
will take all elements of .csv and output final readings for the required values. """

# import necessary libraries
import argparse
import cv2
import os
import numpy as np
import preprocess_image_field as preprocess_image
import pandas as pd
import size_calibration
import time

# shal_lotD4_1550_20191003
# test
#image = ./shal_lotD4_1550_20191003/original_images
#calib = ./calib_lot1_1320_20191003/original_images
#gps = ./shal_lotD4_1550_20191003/shal_lotD4_1550_20191003.csv

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gps filename", required=True, help="Path to GPS file")
ap.add_argument("-c", "--calibration image directory", required=True, help="Path to calibration image directory")
ap.add_argument("-i", "--image directory", required=True, help="Path to image directory")
args = vars(ap.parse_args())

# constants
# Open the GPS file and load the data from it
gps_filename = args["gps filename"]
gps_data = pd.read_csv(gps_filename, sep = ",")
print(gps_data)
print("Number of pictures in file: ", len(gps_data))
headers = list(gps_data)
print("Categories in GPS file: ", headers)

# load the image directory
image_directory = args["image directory"]
print("Images are stored in: ", image_directory)

time.sleep(5)

# image directories
calibration_directory = args["calibration image directory"]

# first perform size calibration to determine the pixel metric
pixel_metric = size_calibration.calibrate(calibration_directory)
print("Calibration complete. \n")
print("Detected pixel metric: ", pixel_metric)

data = []
onion_data = []

for root, dirs, filenames in os.walk(image_directory):
    for i, file in enumerate(sorted(filenames)):

        if i == 260:
            break

        imgpath = os.path.join(root,file) # Reconstructs the file path using
        # the root_directory and currentsfilename
        print("Image %s" % str(i))
        print(imgpath)

        img = cv2.imread(imgpath)
        (total_extra_small, total_small, total_medium, total_large, total_extra_large),\
            wts_result = preprocess_image.preprocess_watershed(img, pixel_metric)
        # print the results for onion count
        #print("Total extra small: ", total_extra_small)
        #print("Total small: ", total_small)
        #print("Total medium: ", total_medium)
        #print("Total large: ", total_large)
        #print("Total extra large: ", total_extra_large)

        # if there is no more GPS data to match the images, break out of the loop
        latitude = gps_data[headers[5]][i]
        longitude = gps_data[headers[6]][i]
        speed = gps_data[headers[7]][i]
        timestamp = gps_data[headers[8]][i]
        result_filename = imgpath
        log = [total_extra_small, total_small, total_medium, total_large, total_extra_large, latitude, longitude, speed, timestamp, file]
        print(log)
        data.append(log)
        cv2.imwrite(file, wts_result)

        cv2.imshow("Segmentation Result", wts_result)
        cv2.waitKey(500)

data = np.array(data)
final_data = pd.DataFrame(data, columns = ['Extra Small Onions (< 1 3/4 in)', 'Small Onions (1 3/4 - 1 13/16 in)','Medium Onions (1 13/16 - 1 7/8 in)', 'Large Onions(1 7/8 in - 2 in)', 'Extra Large Onions (+ 2 in)', 'Latitude', 'Longitude', 'Speed (km/h)', 'Time', 'Filename'])
final_data.sort_values(by=['Time'])
final_data.to_csv("shal_lotD4_1550_20191003_NEW.csv")

data = np.array(data)
print("Final Data Matrix: ",data)
