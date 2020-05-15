"""
Development of a Machine Vision Based Yield Monitor for Shallot Onions
Precision Agriculture and Sensor Systems Group (PASS)
McGill University, Department of Bioresource Engineering 

yield_monitor.py --- This is a yield monitoring program for the masters thesis of 
Amanda Boatswain Jacques. This software detects onion shapes, classifies them by 
size, and then exports them into a .CSV file with GPS data. 
"""

# program Properties 
__author__ = "Amanda Boatswain Jacques"
__version__ = 9.0

# import the necessary python libraries 
from datetime import datetime
import conf
import os
import pandas as pd
import serial
import sys
import time

# computer vision
import cv2
import numpy as np
from preprocess_image_updated_ver5 import preprocess_original, preprocess_watershed, find_onion_contours
import size_calibration
""" Calibrate Monitor """

""" Define Functions and Variables """

filename = "./medium_onions_rectified.txt"
pic_folder = "C:/Users/Amanda/Documents/final results/all_undistorted"

## useful Functions 
def pretty_print(task, msg):    
    # Pretty Print 
    date = datetime.strftime(datetime.now(), '%d/%b/%Y %H:%M:%S')
    print('[%s] %s\t%s' % (date, task, msg))
    
def create_gps_data(gps_filename, picture_folder):
    data = []
    # ================== Load Previously Collected GPS Data ====================================
    print("[INFO] Importing image labels...")
    gps_data = pd.read_csv(gps_filename, sep = "\t", index_col =0)
    print(gps_data)
    headers = list(gps_data)
    print(headers)
    
    headers=list(gps_data)
    
    for i in range(len(gps_data)):
        imgpath = gps_data[headers[4]][i]

        #folder = picture_folder + "/" + imgpath.split("/")[1] + "_undistorted"
        #print("folder", folder)
        new_imgpath = picture_folder +("/") + imgpath.split("/")[-1].split(".png")[0]+ "_undistorted.png"
        print(new_imgpath)
        if os.path.exists(new_imgpath):
            # if the image path exists, load the image and save the new data
            print(new_imgpath)
            img = cv2.imread(new_imgpath)
            #preprocess image and detect onions
            (onions), result = preprocess_watershed(img[15:410, 30:610, :]) 
            #cv2.imshow("original", img)
            #cv2.imshow("result", result)
            #cv2.waitKey(100)
            #print("Onions :", onions)
            speed = gps_data[headers[1]][i]
            longitude = gps_data[headers[2]][i]
            latitude = gps_data[headers[3]][i]
            #print("speed: ", speed)
            #print("longitude: ", longitude)
            #print("latitude: ", latitude)
            
            result_filename = "C:/Users/Amanda/Documents/final results/final_results_watershed_circle/" + imgpath.split("/")[-1].split(".png")[0]+ "_watershed_circle.png"
            #cv2.imshow("Result", result)
            #cv2.waitKey(300)
            #cv2.imwrite(result_filename, result)
            log = [onions[0], onions[1], onions[2], onions[3], onions[4], latitude, longitude, speed, result_filename] 
            print(log)
            print(result_filename)
            data.append(log)  
     
    data = np.array(data)        
    final_data = pd.DataFrame(data, columns = ['1.75 in', '1.8125 in', '1.875 in', '2.00 in', '2.00+ in', 'Latitude', 'Longitude', 'Speed', 'Filename'])
    final_data.to_csv("field_10_results_yield_monitor_02052019.csv")        
    #cv2.destroyAllWindows()
    
    return gps_data
            
gps = create_gps_data(filename, pic_folder)

 