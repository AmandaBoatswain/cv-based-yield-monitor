# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12,  1:00 PM, 2019

@author: Amanda
"""

import serial
import csv

#port ="COM7" # Add to config file
port ="COM5"
# Add to config file

BAUDRATE = 38400 # Add to config file

gps = serial.Serial(port)
gps.baudrate = BAUDRATE
output_filename = "gps_test.csv"
headers = ["latitude", "latitude_char", "longitude", "longitude_char", "time", "speed (km/h)" ]


with open(output_filename, "a") as csvfile:
    # output = csv.writer(csvfile, delimiter = "\t")
    output = csv.writer(csvfile)
    output.writerow(headers)

    while True:
        # flush the gps serial property
        gps.flushInput()

        try:
            line = gps.readline()
            line = line.decode("utf-8")
            gps_data = line.split(",")

            # retrieve only the RMC sentences
            if gps_data[0] == "$GPRMC":
                # only report GPS sentences if an active valid fix was received
                #if gps_data[2] == "A":
                if gps_data[2] == "V":
                    # convert NMEA data to decimal degrees
                    time = float(gps_data[1])-40000
                    latitude = int(gps_data[3][0:2]) + float(gps_data[3][2:])/60
                    latitude = format(latitude, ".5f")
                    latitude_char = gps_data[4]

                    longitude = abs(int(gps_data[5][0:3]) + float(gps_data[5][3:])/60)*(-1)
                    longitude = format(longitude, ".5f")
                    longitude_char = gps_data[6]

                    # convert the speed from knots to km/h
                    speed = gps_data[7]
                    if speed:
                        speed = float(speed)*1.852
                        format(speed, ".4f")
                    else:
                        speed = 0.000

                print('%s %s %s %s %s %s' % (latitude, latitude_char, longitude, longitude_char, time, speed))
                log = [latitude, latitude_char, longitude, longitude_char, time, speed]
                output.writerow(log)
                gps.reset_output_buffer()

        except KeyboardInterrupt:
            gps.close()
            break
