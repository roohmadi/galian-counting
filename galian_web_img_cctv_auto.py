# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:29:54 2023

@author: roohm
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:02:15 2023

@author: MAT-Admin
"""

import tkinter
from tkinter import *
import cv2
import PIL.Image, PIL.ImageTk
import time
from tkinter.filedialog import askopenfilename,asksaveasfile
from tkinter import filedialog, Tk
import requests


from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import cv2
from PIL import Image, ImageTk
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import shutil
from os.path import exists
import math
import time
import datetime
import urllib.request

from tkinter import filedialog as fd
import datetime
from datetime import date


# importing "cmath" for complex number operations
import cmath
from configparser import ConfigParser


import warnings
warnings.filterwarnings("ignore")
running = False
imageVal = False
saveTempImgFlag = False

production = False

global skip_double,sabes_count, batu_count, cntFileSaveSabes, cntFileSaveBatu, PINTU, img_del_date, host, weight_file
global cntdot
cntdot = 0
skip_double = 0
sabes_count =  0
batu_count = 0
cntFileSaveSabes = 0
cntFileSaveBatu = 0
cntFlag = 0
if production:
    OSWindows = False
    host = 'https://produk-inovatif.com/latihan/galian'
else:
    OSWindows = True
    host = 'https://produk-inovatif.com/latihan/galiantes'


from configparser import ConfigParser
config = ConfigParser()


if OSWindows:
    #--Windows
    path_img = os.getcwd()+"\\images\\"
    path_imgTemp = os.getcwd()+"\\tempImg.jpg"
    path_imgTempDetect = os.getcwd()+'\\'+"tempImgDetect.jpg"
    path_imgupl = os.getcwd()+"\\imgupl\\"
else:
    #--Linux
    path_img = os.getcwd()+"/images/"
    path_imgTemp = os.getcwd()+"/tempImg.jpg"
    path_imgTempDetect = os.getcwd()+'/'+"tempImgDetect.jpg"
    path_imgupl = os.getcwd()+"/imgupl/"

print('--------=---------')
print("image saved to " + path_img)
isExistINI = os.path.exists('galianset.ini')
print(isExistINI)
if isExistINI:
    config.read('galianset.ini')
    val_rtsp = config.get('galian', 'rtspset')
    PINTU = config.get('pintu', 'pintuset')
    img_del_date = int(config.get('img_del', 'date_set'))
    weight_file = config.get('weigth_file', 'weigth_set')


else:
    val_rtsp =  ''
#PINTU = 3

# Load YOLOv5 model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#print(model)
#model = torch.hub.load('ultralytics/yolov5', 'custom', 'galian_200epch_1k5.pt')
isExistweight = os.path.exists(weight_file)
if isExistweight:
    model = torch.hub.load('ultralytics/yolov5', 'custom', weight_file)
else:
    print("Weight file not exist")


class App:
    def __init__(self, window, window_title, video_source='rtsp://192.168.0.101/live/ch00_1'):

        self.window = window
        self.window.title(window_title)
        window_height = 800
        window_width = 1300

        screen_width = self.window .winfo_screenwidth()
        screen_height = self.window .winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))


        self.window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))



        self.window.rowconfigure(0, minsize=100, weight=1)
        self.window.columnconfigure(1, minsize=200, weight=1)

        fr_button = Frame(self.window)
        fr_graph = Frame(self.window)
        fr_result = Frame(self.window)

        # -------
        self.btn_rdcsv = Button(fr_button, text="Close CCTV", )
        btn_wrcsv = Button(fr_button, text="Upload Video",command = self.upload_vid_func)
        btn_baseline = Button(fr_button, text="Open CCTV",command = self.cctv_func)
        btn_quit = Button(fr_button, text="Quit", command=self.window.destroy)

        self.btn_rdcsv.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        btn_wrcsv.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        btn_baseline.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        btn_quit.grid(row=3, column=0, sticky="ew", padx=10, pady=10)

        self.btn_rdcsv.config(width=10, height=5)
        btn_wrcsv.config(width=10, height=5)
        btn_baseline.config(width=10, height=5)
        btn_quit.config(width=10, height=5)
        self.btn_rdcsv["state"] = DISABLED

        lbl_obj = Label(fr_button, text="Summary:", font=('Times 14'))
        lbl_obj.grid(row=15, column=0, padx=10, pady=5, sticky="w")

        lbl_mag_min = Label(fr_button, text="Sabes", font=('Times 14'))
        lbl_mag_min.grid(row=16, column=0, padx=10, pady=5, sticky="w")
        lbl_sabes = Label(fr_button, text=": 0 ", font=('Times 14'))
        lbl_sabes.grid(row=16, column=1, padx=5, pady=5, sticky="w")

        lbl_mag_max = Label(fr_button, text="Batu Belah", font=('Times 14'))
        lbl_mag_max.grid(row=17, column=0, padx=10, pady=5, sticky="w")
        lbl_batu = Label(fr_button, text=": 0 ", font=('Times 14'))
        lbl_batu.grid(row=17, column=1, padx=5, pady=5, sticky="w")

        lbl_phase_min = Label(fr_button, text="Pintu", font=('Times 14'))
        lbl_phase_min.grid(row=18, column=0, padx=10, pady=5, sticky="w")
        lbl_cy = Label(fr_button, text=": 0", font=('Times 14'))
        lbl_cy.grid(row=18, column=1, padx=5, pady=5, sticky="w")

        lbl_phase_max = Label(fr_button, text="Object ", font=('Times 14'))
        lbl_phase_max.grid(row=19, column=0, padx=10, pady=5, sticky="w")
        lbl_val = Label(fr_button, text=": None", font=('Times 14'))
        lbl_val.grid(row=19, column=1, padx=5, pady=5, sticky="w")

        lbl_rtsp1 = Label(fr_button, text="Status: ", font=('Times 14'))
        lbl_rtsp1.grid(row=20, column=0, padx=10, pady=5, sticky="w")
        self.lbl_rtsp1val = Label(fr_button, text=": None", font=('Times 14'))
        self.lbl_rtsp1val.grid(row=20, column=1, padx=5, pady=5, sticky="w")

        # textbox
        lbl_rtsp = Label(fr_button, text="RTSP: ", font=('Times 14'))
        # lbl_rtsp.grid(row=25, column=0, padx=10, pady=5, sticky="w")
        input_text = StringVar()

        entry1 = Entry(fr_button, width=25, textvariable=input_text, justify=CENTER)
        entry1.focus_force()
        # entry1.pack(side = TOP, ipadx = 30, ipady = 6)
        entry1.grid(row=26, column=0, padx=5, pady=5, sticky="w")
# -------
        if production:
        #    print("TEESS")
        #else:
            self.video_source = val_rtsp

            # open video source (by default this will try to open the computer webcam)
            self.vid = MyVideoCapture(self.video_source)

            display_frame2 = tkinter.Frame(self.window)
            display_frame2.place(relx=0.6, rely=0.3, width = 800, height = 900, anchor=tkinter.CENTER)
            self.lmain1 = tkinter.Label(display_frame2)
            self.lmain1.place(x = 0, y = 0, width=800, height=900)

            # After it is called once, the update method will be automatically called every delay milliseconds
            self.delay = 10
            self.update()

        # Button that lets the user take a snapshot
        #self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        #self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)

        fr_button.grid(row=0, column=0, sticky="ns")
        fr_graph.grid(row=0, column=1, sticky="nsew")
        fr_result.grid(row=0, column=2, sticky="nsew")

        self.window.mainloop()

    def cctv_close(self):
        if self.vid.isOpened():
            self.vid.Close()
            self.vid.release()

    def cctv_func(self):
        self.btn_rdcsv["state"]=NORMAL
        self.video_source = val_rtsp

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        display_frame2 = tkinter.Frame(self.window)
        display_frame2.place(relx=0.6, rely=0.3, width = 800, height = 900, anchor=tkinter.CENTER)
        self.lmain1 = tkinter.Label(display_frame2)
        self.lmain1.place(x = 0, y = 0, width=800, height=900)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 10
        self.update()

    def upload_vid_func(self):
        print(imageVal)

        def browse_file():
            def run_yolov5_on_video():
                isExistTempImg = os.path.exists(path_imgTemp)
                print(path_imgTemp)
                print(isExistTempImg)
                if isExistTempImg:
                    os.remove("tempImg.jpg")
                isExistTempImgDetect = os.path.exists(path_imgTempDetect)
                if isExistTempImgDetect:
                    os.remove("tempImgDetect.jpg")

                cap = cv2.VideoCapture(file_path)

                display_frame2 = tkinter.Frame(self.window)
                display_frame2.place(relx=0.5, rely=0.3, width = 800, height = 900, anchor=tkinter.CENTER)

                self.lmain1 = tkinter.Label(display_frame2)
                self.lmain1.place(x = 0, y = 0, width=800, height=900)

                def show_frame():
                    global skip_double,sabes_count, batu_count, cntFileSaveBatu, cntFileSaveSabes, cntFlag,cntdot
                    _, frame = cap.read()
                    if frame is None:
                        return
                    else:
                        w, h = frame.shape[1],frame.shape[0]

                    #print(str(w) + " h: " + str(h))

                    current_frame_small = cv2.resize(frame,(0,0),fx=0.35,fy=0.35)
                    results = model(current_frame_small)

                    frame3 = current_frame_small
                    cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
                    img2 = Image.fromarray(cv2image2)
                    imgtk2 = ImageTk.PhotoImage(image=img2)
                    self.lmain1.imgtk = imgtk2
                    self.lmain1.configure(image=imgtk2)

                    self.lmain1.after(1, show_frame)

                #while (cap.isOpened):
                show_frame()
            filename = filedialog.askopenfilename(filetypes=[("video files", "*.*")])
            file_path = os.path.abspath(filename)

            run_yolov5_on_video()

        browse_frame = tkinter.Frame(self.window, bg = "orange")
        browse_frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        browse_button = tkinter.Button(browse_frame, text="Browse", font= ("Rockwell", 20), bg="red", fg="white", command=browse_file)
        browse_button.pack()
        #self.lbl_rtsp1val['text']=  ": DONE..."

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            current_frame_small = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
            frame3 = current_frame_small
            cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
            img2 = Image.fromarray(cv2image2)
            imgtk2 = ImageTk.PhotoImage(image=img2)
            self.lmain1.imgtk = imgtk2
            self.lmain1.configure(image=imgtk2)
            #self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            #self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), "GALIAN C COUNTER. 1.0")