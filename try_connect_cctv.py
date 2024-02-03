# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:02:13 2024

@author: MAT-Admin
"""

#!/usr/local/bin/python3

import cv2
import datetime
import time
import os

from configparser import ConfigParser
config = ConfigParser()
isExistINI = os.path.exists('galianset.ini')
print(isExistINI)
if isExistINI:
    config.read('galianset.ini')
    val_rtsp = config.get('galian', 'rtspset')
    #val_rtsp = 'rtsp://admin:Kk123456@10.243.40.96/live'
    PINTU = config.get('pintu', 'pintuset')
    img_del_date = int(config.get('img_del', 'date_set'))
    weight_file = config.get('weigth_file', 'weigth_set')
    OStype = int(config.get('OS', 'OS_set'))
else:
    val_rtsp =  ''
print("OStype: " +str(OStype))
if OStype:
    OSWindows = True
    host = 'https://produk-inovatif.com/galiantes'
    print("OS Windows")
else:
    OSWindows = False
    production = True
    host = 'https://produk-inovatif.com/latihan/galian'
    print("OS Linux/Debian")
#host = 'https://produk-inovatif.com/galiantes'
print("host: " + host)
current_time = datetime.datetime.now()        
str_tgl = str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day)
if OSWindows:
    #--Windows
    path_img = os.getcwd()+"\\images\\"
    path_imgTemp = os.getcwd()+"\\tempImg.jpg"
    path_imgTempDetect = os.getcwd()+'\\'+"tempImgDetect.jpg"
    path_imgupl = os.getcwd()+"\\imgupl\\"
    path_log = os.getcwd()+"\\log\\"
else:
    #--Linux
    path_img = os.getcwd()+"/images/"
    path_imgTemp = os.getcwd()+"/tempImg.jpg"
    path_imgTempDetect = os.getcwd()+'/'+"tempImgDetect.jpg"
    path_imgupl = os.getcwd()+"/imgupl/"
    path_log = os.getcwd()+"/log/"

fileLog = path_log + str_tgl + '.logU'
print('--------=---------')
print("image saved to " + path_img)
print("cctv: " + val_rtsp)


def reset_attempts():
    return 50


def process_video(attempts):

    while(True):
        (grabbed, frame) = camera.read()

        if not grabbed:
            print("disconnected!")
            camera.release()

            if attempts > 0:
                time.sleep(5)
                return True
            else:
                return False


recall = True
attempts = reset_attempts()

while(recall):
    #camera = cv2.VideoCapture("rtsp://<ip><port>/live0.264")
    camera = cv2.VideoCapture(val_rtsp)

    if camera.isOpened():
        print("[INFO] Camera connected at " +
              datetime.datetime.now().strftime("%m-%d-%Y %I:%M:%S%p"))
        attempts = reset_attempts()
        recall = process_video(attempts)
    else:
        print("Camera not opened " +
              datetime.datetime.now().strftime("%m-%d-%Y %I:%M:%S%p"))
        camera.release()
        attempts -= 1
        print("attempts: " + str(attempts))

        # give the camera some time to recover
        time.sleep(5)
        continue