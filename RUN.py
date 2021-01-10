# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 00:33:14 2018

@author: Amanda
"""
from YieldMonitor import YieldMonitor
import time
import cv2
import argparse


# activate the conda environment
import subprocess
# cd to the correct execution folder
#subprocess.run('cd Desktop/yield_monitor2')
subprocess.run("activate yield_monitor && 'python RUN.py' ", shell = True)

# construct the argument parse and parse the arguments
"""
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True, help="path to the configuration file")
args = vars(ap.parse_args())
"""

YM = YieldMonitor("conf.json")
time.sleep(10)

YM.pretty_print("[INFO] RUNNING", "Yield Monitor initialized!")
YM.calibrate_monitor()

YM.init_gps()
YM.run()
YM.save_log()


YM.close()
