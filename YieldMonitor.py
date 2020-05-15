# -*- coding: utf-8 -*

""" LAST MODIFIED: Monday August 5th, 2019 """ 

## USAGE cd to final_code
# python YieldMonitor_updated.py --conf conf.json 

"""
Development of a Machine Vision Based Yield Monitor for Shallot Onions and Carrot crops
Precision Agriculture and Sensor Systems (PASS) Research Group
McGill University, Department of Bioresource Engineering 

yield_monitor.py --- This is a yield monitoring program for the masters thesis of 
Amanda Boatswain Jacques. This software detects onion and carrot shapes, classifies them by 
size, and then exports them into a .CSV file with GPS data. 
"""

# program Properties 
__author__ = "Amanda Boatswain Jacques"
__version__ = 10.0

# import the necessary python libraries 
from datetime import datetime
import conf
import os
import pandas as pd
import serial
import sys
import time

# computer vision libraries
import cv2
import numpy as np
import preprocess_image 
import size_calibration 

# create yield monitor class 
class YieldMonitor:
    def __init__(self, config):
        # set current file path
        self.current_dir = sys.argv[0]
        
        # load the configuration file
        
        # check to see whether the config file is valid 
        if config is None: 
            raise ValueError
            
        else:    
            self.conf = conf.Conf(config)
            sources = self.conf["camera_sources"]
            
        # initialize the camera     
        for source in sources:
            try: 
                self.camera = cv2.VideoCapture(source)
                
                if self.camera.isOpened():
                    self.pretty_print("[INFO] CAMERA", "OK: Camera successfully located!")
                    self.pretty_print("[INFO] CAMERA", "Camera initialized!")

            except Exception as e:
                self.pretty_print("[ERROR] CAMERA", "Error: %s" % str(e))
                
                # close the program 
                self.pretty_print("[ERROR] SYSTEM", "Shutting down!")
                self.close()
                
        # create a directory for storing the images and .csv
        self.save_path = self.conf["external_drive"]
        date = datetime.strftime(datetime.now(), "%Y%m%d_%H%M" +"/")
        self.image_directory = self.save_path  + date  
        self.result_directory = self.image_directory + "result_images/"
        self.pretty_print("[INFO] IMAGES", "Images will be saved in: " + self.image_directory)
        
        # if not present, make new directories 
        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)
            os.makedirs(self.result_directory)
        
    ### useful Functions 
    def pretty_print(self, task, msg):    
        # Pretty Print 
        date = datetime.strftime(datetime.now(), '%d/%b/%Y %H:%M:%S')
        print('[%s] %s\t%s' % (date, task, msg))

   ### camera Functions     
    def capture_image(self, write=False, ramp_frames = 40):       
        """ Captures a single image from the camera and returns it in PNG format
        read is the easiest way to get a full image out of a VideoCapture object."""
     
        # let the camera stabilize for 40 frames 
        for i in range(ramp_frames):
            try:
                (retval, self.bgr) = self.camera.read()
                
            except Exception as e:
                self.pretty_print("[ERROR] CAMERA", "Error: %s" % str(e))  
                self.close()
        
        if self.bgr is not None:
            cv2.imshow("Captured Image", self.bgr)
            cv2.waitKey(100)
            
            # save the image 
            if write == True:
                date = datetime.strftime(datetime.now(), "%Y%m%d"+"_"+"%H%M%S")
                # add directory here 
                self.filename = self.image_directory + date + ".png"
                cv2.imwrite(self.filename, self.bgr)
            
            else:
                pass

            return self.bgr    
    
    # perform size calibration 
    def calibrate_monitor(self): 
        self.pretty_print("[INFO] SIZE CALIBRATION", "Calibrating monitor...")
        
        try:
            self.pixel_metric = size_calibration.calibrate(self.conf["calibration_directory"])
            
            if self.pixel_metric is not None: 
                self.pretty_print("[INFO] SIZE CALIBRATION", "Calibration completed. Pixel metric is %.3f." % str(self.pixel_metric))
            else:
                self.pretty_print("[ERROR] SIZE CALIBRATION", "Size calibration not completed. Using default value (pixel metric = 1.0")
                self.pixel_metric = 1.00
        
        except Exception as e: 
            self.pretty_print("[ERROR] SIZE Calibration", "Error: %s" % str(e))  
            
        
        return self.pixel_metric
    
    # perform image processing and detect the onions in an image 
    def find_onions(self, write=True):
       (self.small_count, self.medium_count, self.large_count),
       self.result = preprocess_image.preprocess_watershed(self.bgr, self.pixel_metric)
       
       if write == True:
           date = datetime.strftime(datetime.now(), "%Y%m%d"+"_"+"%H%M%S")
           self.result_filename = self.result_directory + date + ".png"
           cv2.imwrite(self.result_filename, self.result)
       
       return(self.small_count, self.medium_count, self.large_count)
        
    def init_gps(self):
        """  Initialize the gps sensor and set the baudrate. """
        COMNUMS = self.conf["gps_ports"]   
        #self.gps = serial.Serial()
        self.pretty_print("[INFO] GPS", "Initializing GPS... ")
        
        # detect the active gps port and save it 
        for port in COMNUMS:
            try:
                 self.gps = serial.Serial(port, timeout = 0.2)
                 self.gps_port = port
                 # explicit close 'cause of delayed GC in java
                 #self.gps.close()  
            
            except serial.SerialException:
                pass
        
        if self.gps.isOpen():
            # set the gps baudrate
            self.gps.baudrate = self.conf["gps_baudrate"]
            self.pretty_print("[INFO] GPS", "GPS at port %s with baud %s! " % (self.gps_port, self.gps.baudrate))
        
        else:
            self.pretty_print("[ERROR] GPS", "GPS not found!.")
            # close the GPS serial object 
            self.close()
            
            
    def get_position(self):
        """ Get the current position (latitude, longitude, speed) of the image. """
        # retrieve only the RMC sentences 
        code = "RMC"
        
        while True: 
            try:
                line = self.gps.readline()
                line = line.decode("utf-8")
                #print(line)
                
                if line.find(code) > 0:
                    break
    
            except UnicodeDecodeError:
                pass

        gps_data = line.split(",")
        
        # only report GPS sentences if an active valid fix was received
        if gps_data[2] == "A":
            self.pretty_print("[INFO] GPS", "Active GPS Fix found!")
            self.latitude = gps_data[3] 
            self.latitude_char = gps_data[4]        
            self.longitude = gps_data[5]
            self.longitude_char = gps_data[6]        
            self.speed = gps_data[7]
            
        else: 
            self.pretty_print("[ERROR] GPS", "No active GPS fix was found! Position data is not valid.")
            self.latitude = [] 
            self.latitude_char = []        
            self.longitude = []
            self.longitude_char = []        
            self.speed = []
            
        if self.speed is not None:
            # convert the gps speed (knots) to km/h  
            self.speed = float(gps_data[7])*1.852
            self.speed = format(self.speed, ".3f") 
        else:
            pass
                
        return (self.latitude, self.longitude, self.speed)
    
    def run(self):
        """ Run the program continuously. Get captures, 
        analyze them, and then save the current position. """
         
        self.pretty_print("[INFO] RUNNING", "Yield Monitor initialized!")
        self.pretty_print("[INFO] RUNNING", "Running yield monitoring program. Press CTRL+C to exit.")
        
        # open the GPS port, give some time for GPS and camera to stabilize 
        time.sleep(5)
        
        # initialize the data matrix 
        self.data = []
        
        while (True):
            try:
                self.capture_image(write =True)
                small, medium, large = self.find_onions()
                lat, long, speed = self.get_position()
                self.log = [small, medium, large, lat, long, speed] 
                cv2.imshow("result", self.result)
                cv2.waitKey(100)
                
                if self.filename is not None:
                    self.log = [small, medium, large, lat, long, speed, self.result_filename]    
                    
                self.data.append(self.log)
                
                print(self.log)
            
            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                self.gps.close()
                self.data = np.array(self.data)
                
                if self.filename is not None:
                    self.dataframe = pd.DataFrame(self.data, columns = ['Small Onions','Medium Onions', 'Large Onions', 'Latitude', 'Longitude', 'Speed', 'Filename'])
                
                else: 
                    self.dataframe = pd.DataFrame(self.data, columns = ['Small Onions', 'Medium Onions', 'Large Onions', 'Latitude', 'Longitude', 'Speed'])
                
                # Emergency save the data that was collected until this point under a default name 
                self.dataframe.to_csv(self.conf["log_file_path"] + "defland_test" + ".csv")
                
                break
        
        return self.df
    
    ### write everything to a csv
    def save_log(self):
        time.sleep(2)
        test_filename = input("Please enter the name of the results file (use only numbers, letters and underscores):  " )
        self.pretty_print("[INFO] SAVING", "Saving results from test run as %s " % (test_filename + ".csv"))
        self.dataframe.to_csv(self.conf["log_file_path"] + test_filename + ".csv")
            
    ### close the program
    def close(self):
        # shut down the program and delete camera and GPS source 
        self.pretty_print("[INFO] WARN", "Shutdown triggered!")
        time.sleep(5)
        self.gps.close()
        self.camera.release()
        cv2.destroyAllWindows()
        
        sys.exit()

