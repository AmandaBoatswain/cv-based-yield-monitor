# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 00:33:14 2019

@author: Amanda

Development of a Machine Vision Based Yield Monitor for Shallot Onions and Carrot crops
Precision Agriculture and Sensor Systems (PASS) Research Group
McGill University, Department of Bioresource Engineering

RUN.py --- This is a yield monitoring program for the masters thesis of
Amanda Boatswain Jacques. This is the main operation file. 
"""
from YieldMonitor import YieldMonitor
import time
import cv2
import argparse


# activate the conda environment
import subprocess

# cd to the correct execution folder
# >> cd Desktop/yield_monitor

# This next line of code opens the "yield_monitor" conda env and runs the main file "RUN.py"
# when you click on the file icon
subprocess.run("activate yield_monitor && 'python RUN.py' ", shell = True)

# Execute the yield monitoring program and initialize the camera source
YM = YieldMonitor("conf.json")
time.sleep(5)
YM.pretty_print("[INFO] RUNNING", "Yield Monitor initialized!")

# Perform size calibration
YM.calibrate_monitor()

# Initialize the GPS sensor
YM.init_gps()

# Run the program, to exit press "Ctrl + C"
YM.run()

# Save the data to disk
YM.save_log()

# Close the program
YM.close()
