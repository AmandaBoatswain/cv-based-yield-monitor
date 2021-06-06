# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 19:16:24 2021

@author: amand
"""

# Import librairies
import os 
import cv2
import cluster_k_means

# Define the working directory
directory = "training"

# crawling through directory and subdirectories
for root, directories, files in os.walk(directory):
    for filename in files:
        # join the two strings in order to form the full filepath.
        filepath = os.path.join(root, filename)
        
        # don't open desktop.ini file
        if filepath.find(".ini") > -1:
                pass
        
        else:
            print(filepath)
            img = cv2.imread(filepath)        
            result = cluster_k_means.preprocess_watershed(img)
            cv2.imshow("Result", result)
        
### loop through the images and run the K-Means clustering algorithm