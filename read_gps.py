# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 17:51:04 2018

@author: Amanda
"""

import serial

port ="COM7" # Add to config file
BAUDRATE = 38400 # Add to config file

gps = serial.Serial(port)
gps.baudrate = BAUDRATE

while True:
    # flush the gps serial port
    #gps.flushInput()
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
                latitude = gps_data[3])
                latitude = int(latitude[0:2] + float(latitude[3:])/60
                latitude_char = gps_data[4]

                longitude = gps_data[5]
                longitude = abs(int(latitude[0:3]) + float(latitude[4:])/60)*(-1)
                longitude_char = gps_data[6]
                speed = gps_data[7]
                #maybe try changing speed to m/s tomorrow

                """
                if speed:
                    speed = float(gps_data[7]*1.852
                """
                #speed = format(speed, ".3f")
                #speed = str(speed)

                print('%s %s %s %s %s %s' % (latitude, latitude_char, longitude, longitude_char, speed, "km/h" ))

                print('%s %s %s %s ' % (latitude, latitude_char, longitude, longitude_char))

    except KeyboardInterrupt:
        gps.close()
        break
