# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:02:13 2024

@author: MAT-Admin
"""

#!/usr/local/bin/python3

import cv2
import datetime
import time

val_rtsp = 'rtsp://admin:Kk123456@192.168.0.101/live'

def reset_attempts():
    return 50


def process_video(attempts):

    while(True):
        (grabbed, frame) = camera.read()

        if not grabbed:
            print("disconnected!")
            camera.release()

            if attempts > 0:
                time.sleep(2)
                return True
            else:
                return False


recall = True
attempts = reset_attempts()

while(recall):
    #camera = cv2.VideoCapture("rtsp://<ip><port>/live0.264")
    print(val_rtsp)
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
        time.sleep(2)
        continue