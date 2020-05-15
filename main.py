# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 00:33:14 2018

@author: Amanda
"""
import YieldMonitor 
import time 
import argparse

# construct the argument parse and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True, help="path to the configuration file")
args = vars(ap.parse_args())

YM = YieldMonitor.YieldMonitor(args["config"])
time.sleep(2)

YM.pretty_print("[INFO] RUNNING", "Yield Monitor initialized!")
YM.calibrate_monitor()
time.sleep(2)
YM.init_gps()

YM.run()
YM.save_log()


YM.close()