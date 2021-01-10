# -*- coding: utf-8 -*

""" LAST MODIFIED: Friday September 13th, 2019 """

## USAGE cd to Desktop/yield_monitor
# python RUN.py

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
__version__ = 10.5

# import the necessary python libraries
from datetime import datetime
import conf
import csv
import os
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"
import pandas as pd
import serial
import sys
import time

# computer vision libraries
import cv2
import numpy as np
import preprocess_image
import size_calibration

#------------------------INITIAL SETUP------------------------ #
class YieldMonitor:
    """ create yield monitor class. """
    def __init__(self, config):
        # load the configuration file and check to see whether the config file is valid
        if config is None:
            raise ValueError

        else:
            self.conf = conf.Conf(config)
            sources = self.conf["camera_sources"]

        # initialize the camera
        for source in sources:
            try:
                self.camera = cv2.VideoCapture(source, cv2.CAP_DSHOW)

                if self.camera.isOpened():
                    self.pretty_print("[INFO] CAMERA", "Camera successfully located!")
                    self.pretty_print("[INFO] CAMERA", "Camera initialized!")

            except Exception as e:
                self.pretty_print("[ERROR] CAMERA", "Error: %s" % str(e))
                # close the program
                self.pretty_print("[ERROR] SYSTEM", "Shutting down!")
                self.close()

        # get harvest info - append the date - HourMin_YearMonthDay
        self.current_vegetable = input("Please enter the name of the vegetable being harvested: ")
        self.current_lot = input("Please enter the lot number: ")
        date = datetime.strftime(datetime.now(), "%H%M"+"_"+"%Y%m%d")

        # create directories to store the images and output file
        self.output_directory = self.conf["external_drive"] + self.current_vegetable + "_" + "lot" + self.current_lot + "_"+ date
        self.original_image_directory = self.output_directory + "/original_images/"
        self.result_image_directory = self.output_directory + "/result_images/"
        self.output_filename = self.output_directory + "/" + self.current_vegetable + "_" + "lot" + self.current_lot + "_"+ date + ".csv"

        # if not present, make new directories
        try:
            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)
                os.makedirs(self.original_image_directory)
                os.makedirs(self.result_image_directory)

        except Exception as e:
            self.pretty_print("[ERROR] EXTERNAL DRIVE", "External drive not found: %s." % str(e))
            # close the program
            self.pretty_print("[ERROR] SYSTEM", "Please check connection and try again.")
            self.close()


        self.pretty_print("[INFO] DATA", "Result images and .csv will be saved in: " + self.output_directory)

        # make the output file if it doesn't exist
        try:
            self.output_file = open(self.output_filename, "x")
            self.output_file.close()

        except Exception as e:
            self.pretty_print("[ERROR] OUTPUT FILE", "That file name already exists. Please wait for a few seconds and retry. %s" % str(e))

    ### useful Functions
    def pretty_print(self, task, msg):
        # Pretty Print
        date = datetime.strftime(datetime.now(), '%d/%b/%Y %H:%M:%S')
        print('[%s] %s\t%s' % (date, task, msg))

   #------------------------CAMERA SETUP------------------------ #
   ### camera Functions
    def capture_image(self, write=True, ramp_frames = 20):
        """ Captures a single image from the camera and returns it in PNG format
        read is the easiest way to get a full image out of a VideoCapture object."""

        # let the camera stabilize for 40 frames
        for i in range(ramp_frames):
            try:
                (retval, self.original_image) = self.camera.read()

            except Exception as e:
                self.pretty_print("[ERROR] CAMERA", "Error: %s" % str(e))
                self.close()

        if self.original_image is not None:
            # save the image
            if write == True:
                self.date = datetime.strftime(datetime.now(),"%H%M%S")
                self.original_image_filename = self.original_image_directory + self.date + ".png"
                cv2.imwrite(self.original_image_filename, self.original_image)

            else:
                pass

            return self.original_image

    #------------------------GPS SETUP------------------------ #
    def init_gps(self):
        """  Initialize the gps sensor and set the baudrate. """
        COMNUMS = self.conf["gps_ports"]
        self.pretty_print("[INFO] GPS", "Initializing GPS... ")

        # detect the active gps port and save it
        for port in COMNUMS:
            try:
                 self.gps = serial.Serial(port, timeout = 4.0)
                 self.gps_port = port

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
        while True:
            try:
                line = self.gps.readline()
                line = line.decode("utf-8")
                #print(line)

                if line.find("RMC") > 0:
                    break

            except UnicodeDecodeError:
                pass

        gps_data = line.split(",")

        # GPS sentences will be accurate if an active "A" valid fix was received
        if gps_data[2] == "A":
            self.pretty_print("[INFO] GPS", "Active GPS Fix found!")
            # convert NMEA data to decimal degrees
            self.gps_time = float(gps_data[1])-40000
            self.latitude = int(gps_data[3][0:2]) + float(gps_data[3][2:])/60
            self.latitude = format(self.latitude, ".5f")
            self.latitude_char = gps_data[4]

            self.longitude = abs(int(gps_data[5][0:3]) + float(gps_data[5][3:])/60)*(-1)
            self.longitude = format(self.longitude, ".5f")
            self.longitude_char = gps_data[6]

            # convert the speed from knots to km/h
            self.speed = gps_data[7]
            if self.speed:
                self.speed = float(self.speed)*1.852
                format(self.speed, ".3f")
            else:
                self.speed = 0.000

        else:
            self.pretty_print("[ERROR] GPS", "No active GPS fix was found! Position data is not valid.")
            self.gps_time = float(gps_data[1])-40000
            self.latitude = []
            self.latitude_char = []
            self.longitude = []
            self.longitude_char = []
            self.speed = []

        # reset the gps port for new data reading
        self.gps.flushInput()
        self.gps.reset_output_buffer()

        return (self.latitude, self.longitude, self.speed, self.gps_time)

    #------------------------SIZE CALIBRATION------------------------ #
    def calibrate_monitor(self):
        """ perform size calibration. """
        self.pretty_print("[INFO] SIZE CALIBRATION", "Calibrating monitor...")

        try:
            self.pretty_print("[INFO] SIZE CALIBRATION", "Loading images from %s." % str(self.conf["calibration_directory"]))
            self.pixel_metric = size_calibration.calibrate(self.conf["calibration_directory"])

            if self.pixel_metric:
                self.pretty_print("[INFO] SIZE CALIBRATION", "Calibration completed. Pixel metric is %.3f." % self.pixel_metric)

        except Exception as e:
                self.pretty_print("[ERROR] SIZE Calibration", "Error: %s" % str(e))
                # set a default pixel metric value
                self.pixel_metric = 3.380 # default value calculated during previous testing
                self.pretty_print("[ERROR] SIZE CALIBRATION", "Size calibration not completed. Using default value (pixel metric = %.3f)" % self.pixel_metric)

        return self.pixel_metric

    #------------------------ONION DETECTION------------------------ #
    def find_onions(self, write=True):
        """ perform image processing and detect the onions in an image."""
        self.onions, self.result_image = preprocess_image.preprocess_watershed(self.original_image, self.pixel_metric)
        self.extra_small_count = self.onions[0]
        self.small_count = self.onions[1]
        self.medium_count = self.onions[2]
        self.large_count = self.onions[3]
        self.extra_large_count = self.onions[4]

        if write == True:
            #date = datetime.strftime(datetime.now(), "%H%M%S")
            self.result_filename = self.result_image_directory + self.date + ".png"
            cv2.imwrite(self.result_filename, self.result_image)

        return(self.extra_small_count, self.small_count, self.medium_count, self.large_count, self.extra_large_count)

    #------------------------PROGRAM RUNNING------------------------ #
    def run(self):
        """ Run the program continuously. Get captures,
        analyze them, and then save the current position. """

        self.pretty_print("[INFO] RUNNING", "Yield Monitor initialized!")
        self.pretty_print("[INFO] RUNNING", "Running yield monitoring program. Press CTRL+C to exit.")

        # open the GPS port, give some time for GPS and camera to stabilize
        time.sleep(2)

        # initialize the data matrix
        self.data = []
        headers = ['Extra Small Onions (< 1 3/4 in)', 'Small Onions (1 3/4 - 1 13/16 in)','Medium Onions (1 13/16 - 1 7/8 in)', 'Large Onions(1 7/8 in - 2 in)', 'Extra Large Onions (+ 2 in)', 'Latitude', 'Longitude', 'Speed (km/h)', 'Time', 'Filename']

        # start writing the data to a csv file
        with open(self.output_filename, "a", newline ='') as csvfile:
            output = csv.writer(csvfile)
            output.writerow(headers)
            while (True):
                try:
                    # capture an image and get coordinates
                    lat, long, speed, gps_time = self.get_position()
                    self.capture_image(write=True)
                    time.sleep(0.5)

                    # start image processing
                    extra_small, small, medium, large, extra_large = self.find_onions()
                    current_time = datetime.strftime(datetime.now(), "%H%M%S")
                    self.log = [extra_small, small, medium, large, extra_large, lat, long, speed, gps_time]
                    cv2.imshow("result", self.result_image)
                    cv2.waitKey(500)

                    if self.result_image is not None:
                        self.log = [extra_small, small, medium, large, extra_large, lat, long, speed, gps_time, self.result_filename]
                        output.writerow(self.log)
                        print(self.log)

                except KeyboardInterrupt:
                    print("Keyboard Interrupt has been caught!")
                    cv2.destroyAllWindows()
                    break

    ### write everything to a csv
    def save_log(self):
        time.sleep(2)
        self.pretty_print("[INFO] SAVING", "Saving results from test run as %s " % (self.output_filename))

    ### close the program
    def close(self):
        # shut down the program and delete camera and GPS source
        self.pretty_print("[INFO] WARN", "Shutdown triggered! Closing camera and GPS sensor.")
        time.sleep(5)
        self.gps.close()
        self.camera.release()
        cv2.destroyAllWindows()
        sys.exit()
