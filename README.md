# Development of a Machine Vision-Based Yield Monitoring System 

This code was developed as a requirement for my Masters thesis project. Crop yield estimation and mapping are important tools that can help growers efficiently use their available resources and have access to detailed representations of their farm. 

## Project Abstract
Technical advancements in computer and machine vision have improved the detection, quality assessment and yield estimation processes for crops including apples, citrus, mangoes, maize, figs and many other fruits. However, similar methods capable of exporting a detailed yield map for vegetable crops have yet to be fully developed. A machine vision-based yield monitor was designed to perform identification, size categorization and continuous counting of shallot onions in-situ during the harvesting process. The system is composed of a video logger and global positioning system (GPS), coupled with computer software developed in Python. Computer vision analysis is performed within the tractor itself while an RGB camera positioned directly above the harvesting conveyor collects real time video data of the crops under natural sunlight conditions. Vegetables are segmented using Watershed segmentation, detected on the conveyor and then classified by size. Results showed that the system was able to correctly detect 62.6% of onions in a subsample of the dataset and showed a correlation coefficient (R2) of 0.494 between true and estimated counts. The software was also evaluated on its ability to classify the onions into 3 size categories (small, medium and large). A total of 55.9% of 271 analyzed onions were correctly categorized, with the highest performance achieved in the large class (73.3%), followed by the small class (58.7%) and medium class (44.4%). Based on the obtained results, occasional occlusion of vegetables and inconsistent lighting conditions were the main factors that inhibited performance.  Finally, these geotagged images were used to map the size distribution of the shallot onions on a small section of the onion field. Although further enhancements are envisioned for the prototype system to improve overall detection and size classification, its modular and novel design allows it to be used to map a selection of crops including carrots, shallot onions, Chinese radish and lettuce crops. The system has the potential to benefit many producers of small vegetable crops by providing them with useful harvest information in real time that can significantly improve current harvesting logistics. 

## Configuration Requirements
To run this code, you need to have a version of python 3 up and running. This code was tested using python version 3.8.3. Python libraries including scikit-image, scikit-learn, scipy, imutils, numpy, pyserial, subprocess, cv2 and pandas must be installed in your python environment. You will also need to have a usb camera and GPS sensor connected to the computer you are running it from. The GPS sensor will need to be accessed using the ` ` `serial.Serial()` ` ` command in the pyserial library. Please make sure to update the conf.json file with the correct location of the camera source and GPS sensor ports. If saving images is necessary, please make sure to change the "external_drive"
location in the conf.json file as well.

### Python version
* [python 3.8.3](https://www.python.org/downloads/release/python-383/)

### Running this Code
Download all the files from the zip folder in this repository. Open a terminal and cd to the project folder (if you are using a python environment, make sure to activate it before). You can also run the code by clicking on the RUN.py file if the subprocess library is installed. 

` ` `https://github.com/AmandaBoatswain/cv-based-yield-monitor` ` ` 

## Helper Functions
Helper functions were created to help check if the peripheral hardware is properly connected. The ` ` `camera_test.py` ` `  file checks if the USB camera can be accessed and shows the camera feed on the monitor, and the ` ` `read_gps` ` `  and ` ` `read_gps_new` ` ` files test the GPS sensor. A [GARMIN 19X HVS sensor](https://buy.garmin.com/en-CA/CA/p/100686) was used as a GPS during testing and was connected to the computer using a standard RS-232 serial connector.  

## Contact information
For any questions or concerns regarding this tutorial, please contact amanda.boatswainj@gmail.com



