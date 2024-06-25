# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 14:10:23 2024

@author: MAT-Admin
"""

import tkinter as tk
from tkinter import *
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

from tkinter import filedialog as fd
import datetime
from datetime import date
import urllib.request
import os
import torch
import shutil
from os.path import exists
import requests

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best_plate.pt')


root = Tk()
#root = Toplevel()
root.title("GALIAN C COUNTER. 1.0")
#window_height = 800
#window_width = 1280
#window_height = 460
#window_width = 800
window_height = 800
window_width = 1200

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))

root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))



root.rowconfigure(0, minsize=100, weight=1)
root.columnconfigure(1, minsize=200, weight=1)

fr_button = Frame(root)
fr_graph = Frame(root)
fr_result = Frame(root)

#================== begin GUI ========

fig = Figure(figsize = (6, 5), dpi = 100)

def open_image():
    def browse_file():
        def run_yolov5_on_video():


            width, height = 1200, 1200
            cap = cv2.VideoCapture(file_path)
            #cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


            display_frame2 = tk.Frame(root)
            display_frame2.place(relx=0.5, rely=0.3, width = 800, height = 900, anchor=tk.CENTER)


            lmain1 = tk.Label(display_frame2)
            lmain1.place(x = 0, y = 0, width=800, height=900)

            def show_frame():
                    global skip_double,sabes_count, batu_count, cntFileSaveBatu, cntFileSaveSabes,cntdot
                    _, frame = cap.read()
                    if frame is None:
                        return
                    else:
                        w, h = frame.shape[1],frame.shape[0]
                        current_frame_small = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)

                        #print(str(w) + " h: " + str(h))
                        

                        results = model(current_frame_small)
                        for *xyxy, conf, cls in results.xyxy[0]:
                            #cv2.line(frame, (0, h-400), (w, h-400), (0,255,0), thickness=3)
                            if conf>0.8:
                                label = f'{model.names[int(cls)]} {conf:.2f}'

                                cv2.rectangle(current_frame_small, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,0,255), 2)
                                cv2.putText(current_frame_small, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                            
                           # cv2.putText(current_frame_small,  "Process time: " + str(round(elapsed,2)) + " s/frame",(450,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
                        


                        frame3 = current_frame_small
                        cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
                        img2 = Image.fromarray(cv2image2)

                        imgtk2 = ImageTk.PhotoImage(image=img2)

                        lmain1.imgtk = imgtk2
                        lmain1.configure(image=imgtk2)
                    
                    lmain1.after(1, show_frame)
            #while (cap.isOpened):
            show_frame()

        filename = filedialog.askopenfilename(filetypes=[("video files", "*.*")])
        file_path = os.path.abspath(filename)

        run_yolov5_on_video()
   
    #main_frame.place_forget()

    browse_frame = tk.Frame(root, bg = "orange")
    browse_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
   
    browse_button = tk.Button(browse_frame, text="Browse", font= ("Rockwell", 20), bg="Yellow", fg="white", command=browse_file)
    browse_button.pack()
    

def Start(label):
	global running
	running=True
	counter_label(label)
	btn_auto["state"] = DISABLED
	btn_stop["state"]= NORMAL
	btn_Single["state"] = DISABLED
	btn_rdcsv["state"] = DISABLED
	btn_wrcsv["state"] = DISABLED
	
# Stop function of the stopwatch
def Stop():
	global running
	running = False
	btn_stop["state"]=DISABLED
	btn_auto["state"] = NORMAL
	btn_Single["state"] = NORMAL
	btn_rdcsv["state"] = NORMAL
	btn_wrcsv["state"] = NORMAL
#----end function ------

btn_rdcsv = Button(fr_button, text="Cek Image", command=open_image)
btn_wrcsv = Button(fr_button, text="Upload Video", command=open_image)
btn_baseline = Button(fr_button, text="CCTV",command=open_image)
btn_quit = Button(fr_button, text="Quit", command=root.destroy)

btn_rdcsv.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
btn_wrcsv.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
btn_baseline.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
btn_quit.grid(row=3, column=0, sticky="ew", padx=10, pady=10)

btn_rdcsv.config(width=10, height=5)
btn_wrcsv.config(width=10, height=5)
btn_baseline.config(width=10, height=5)
btn_quit.config(width=10, height=5)
#-------------------------------------


#-------------------------------------

#my_logo.grid(row=6, column=2)
#brin = ImageTk.PhotoImage(Image.open("brin50.png")) # load file
#my_logo = Label(fr_button, image=brin)
#my_logo.grid(row=6, column=0)

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

lbl_phase_min = Label(fr_button, text="Cy", font=('Times 14'))
lbl_phase_min.grid(row=18, column=0, padx=10, pady=5, sticky="w")
lbl_cy = Label(fr_button, text=": 0", font=('Times 14'))
lbl_cy.grid(row=18, column=1, padx=5, pady=5, sticky="w")

lbl_phase_max = Label(fr_button, text="val", font=('Times 14'))
lbl_phase_max.grid(row=19, column=0, padx=10, pady=5, sticky="w")
lbl_val = Label(fr_button, text=": 0", font=('Times 14'))
lbl_val.grid(row=19, column=1, padx=5, pady=5, sticky="w")

# textbox
lbl_rtsp = Label(fr_button, text="RTSP: ", font=('Times 14'))
#lbl_rtsp.grid(row=25, column=0, padx=10, pady=5, sticky="w")
input_text = StringVar() 
  
entry1 = Entry(fr_button, width=25, textvariable = input_text, justify = CENTER)
entry1.focus_force() 
#entry1.pack(side = TOP, ipadx = 30, ipady = 6)
#entry1.grid(row=26, column=0, padx=5, pady=5, sticky="w")
#-------------------------------------
result_label1 = Label(fr_result, text="Crack Estimation:                            ", font=("Bold"))
#================== end GUI ==========
fr_button.grid(row=0, column=0, sticky="ns")
fr_graph.grid(row=0, column=1, sticky="nsew")
fr_result.grid(row=0, column=2, sticky="nsew")

from configparser import ConfigParser
config = ConfigParser()

config.read('galianset.ini')
val_rtsp = config.get('galian', 'rtspset')
entry1.insert(0, val_rtsp)


root.mainloop()