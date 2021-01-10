# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 17:51:04 2018

@author: Amanda-
"""

import serial

#port ="COM7" # Add to config file
port ="COM5"
# Add to config file

BAUDRATE = 38400 # Add to config file

gps = serial.Serial(port)
gps.baudrate = BAUDRATE


while True:
    # flush the gps serial port
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

                print('%s %s %s %s %s %s' % (latitude, latitude_char, longitude, longitude_char, speed, "km/h" ))

        gps.reset_output_buffer()

    except KeyboardInterrupt:
        gps.close()
        break
