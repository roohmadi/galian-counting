# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:14:16 2023

@author: roohm
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 13:07:37 2023

@author: roohm
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 20:12:46 2023

@author: capac
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
import datetime
import urllib.request

from tkinter import filedialog as fd

# from yolov5.models.experimental import attempt_load
# from yolov5.utils.downloads import attempt_download
# from yolov5.models.common import DetectMultiBackend
# from yolov5.utils.dataloaders import LoadImages, LoadStreams
# from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, 
#                                   check_imshow, xyxy2xywh, increment_path)
# from yolov5.utils.torch_utils import select_device, time_sync
# from yolov5.utils.plots import Annotator, colors
# from deep_sort.utils.parser import get_config
# from deep_sort.deep_sort import DeepSort

# Python code to demonstrate the working of
# complex(), real() and imag()
  
# importing "cmath" for complex number operations
import cmath
from configparser import ConfigParser


import warnings
warnings.filterwarnings("ignore")
running = False
imageVal = False
saveTempImgFlag = False



global skip_double,sabes_count, batu_count, cntFileSaveSabes, cntFileSaveBatu, PINTU
skip_double = 0
sabes_count =  0
batu_count = 0
cntFileSaveSabes = 0
cntFileSaveBatu = 0
cntFlag = 0

from configparser import ConfigParser
config = ConfigParser()

print('--------=---------')
isExistINI = os.path.exists('galianset.ini')
print(isExistINI)
if isExistINI:
    config.read('galianset.ini')
    val_rtsp = config.get('galian', 'rtspset')
    PINTU = config.get('pintu', 'pintuset')
else:
    val_rtsp =  ''
#PINTU = 3

# Load YOLOv5 model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#print(model)
model = torch.hub.load('ultralytics/yolov5', 'custom', 'galian_200epch_1k5.pt')

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

def connect(host='http://google.com'):
    try:
        urllib.request.urlopen(host) #Python 3.x
        return True
    except:
        return False

def web_cam_func():
    width, height = 400, 400
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


    display_frame2 = tk.Frame(root)
    display_frame2.place(relx=0.5, rely=0.3, width = 600, height = 700, anchor=tk.CENTER)


    lmain1 = tk.Label(display_frame2)
    lmain1.place(x = 0, y = 100, width=600, height=600)

    def show_frame():
            _, frame = cap.read()
            # Perform inference
            results = model(frame)
            w, h = frame.shape[1],frame.shape[0]
            #print(str(w) + " h: " + str(h))

             # Parse results and draw bounding boxes
            for *xyxy, conf, cls in results.xyxy[0]:
                cv2.line(frame, (0, h-400), (w, h-400), (0,255,0), thickness=3)
                if conf>0.8:
                    label = f'{model.names[int(cls)]} {conf:.2f}'

                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,0,255), 2)
                    cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            #frame3 = cv2.flip(frame, 1)
            frame3 = frame
            cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
            img2 = Image.fromarray(cv2image2)

            imgtk2 = ImageTk.PhotoImage(image=img2)

            lmain1.imgtk = imgtk2
            lmain1.configure(image=imgtk2)
            
            lmain1.after(10, show_frame)
        
    show_frame()

def get_location():
    import socket
    import requests
    from ip2geotools.databases.noncommercial import DbIpCity
    from geopy.distance import distance

    def printDetails(ip):
        res = DbIpCity.get(ip, api_key="free")
        print(f"IP Address: {res.ip_address}")
        print(f"Location: {res.city}, {res.region}, {res.country}")
        print(f"Coordinates: (Lat: {res.latitude}, Lng: {res.longitude})")

    #print(requests.get('http://ip.42.pl/raw').text)

    ip_add = requests.get('http://ip.42.pl/raw').text#input("Enter IP: ")  # 198.35.26.96
    printDetails(ip_add)


def cctv_func123():
    global sabes_count
    sabes_count += 1
    muatan = 1
    # using now() to get current time
    current_time = datetime.datetime.now()
    tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
    jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)
    print(tgl)
    print(jam)
    image_file = "Img_" + str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) +"_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second) + "_" + str(muatan) + ".jpg"
    img_file = 'Img_2023_10_27_22_45_26_1.jpg'
    print(image_file)

    url = 'https://produk-inovatif.com/latihan/galian/galian.php?ins=2'
    #data = { 'tgl': '2023-10-26', 'jam': '21:31:00', 'muatan': '1', 'pintu': '4', 'filename': 'test.jpg', 'source': 'upload'}
    data = { 'tgl': tgl, 'jam': jam, 'muatan': '1', 'pintu': '4', 'filename': img_file, 'source': 'upload'}

    head = {'Content-Type' : 'application/x-www-form-urlencoded'}
    x = requests.post(url, data=data, headers=head)


    print(x.text)
    url = 'https://produk-inovatif.com/latihan/galian/galian.php?pintu=2'
    #data = {  'sabes': '2', 'batubelah': '0', 'pintu': '2'}
    data = {  'sabes': '2', 'batubelah': '0', 'pintu': '2'}

    head = {'Content-Type' : 'application/x-www-form-urlencoded'}
    x = requests.post(url, data=data, headers=head)

    print(x.text)


    dfile = open(img_file, "rb").read()

    url_img = 'https://produk-inovatif.com/latihan/galian/img_py.php'
    files= {'file': (img_file,dfile,'image/jpg',{'Expires': '0'}) }
    test_res = requests.post(url_img, files =  files)
    print(test_res)


def cctv_func():
    isExistTempImg = os.path.exists(os.getcwd()+'\\'+"tempImg.jpg")
    print(os.getcwd()+'\\'+"tempImg.jpg")
    print(isExistTempImg)
    if isExistTempImg:
        os.remove("tempImg.jpg")
    isExistTempImgDetect = os.path.exists(os.getcwd()+'\\'+"tempImgDetect.jpg")
    if isExistTempImgDetect:
        os.remove("tempImgDetect.jpg")

    #cap = cv2.VideoCapture('C:\\Users\\roohm\\vehicle-counting-yolov5\\data\\video\\sabesdouble.mp4')
    #cap = cv2.VideoCapture('E:\\YOLO\\counting\\vechicle-counting-yolo-main\\Crash Kalteng3.mp4')
    cap = cv2.VideoCapture(val_rtsp)

    display_frame2 = tk.Frame(root)
    display_frame2.place(relx=0.5, rely=0.3, width = 800, height = 900, anchor=tk.CENTER)


    lmain1 = tk.Label(display_frame2)
    lmain1.place(x = 0, y = 0, width=800, height=900)

    def show_frame():
            global skip_double,sabes_count, batu_count, cntFileSaveBatu, cntFileSaveSabes, cntFlag
            _, frame = cap.read()
            if frame is None:
                return
            else:
                w, h = frame.shape[1],frame.shape[0]
            current_frame_small = cv2.resize(frame,(0,0),fx=0.35,fy=0.35)
            # Perform inference
            results = model(current_frame_small)
            print(".")
            lbl_val['text']=  "...."
            reUPLOAD_img()

            if cntFlag==0:
                cv2.imwrite("tempImgDetect.jpg", current_frame_small)
                cntFlag = 1

             # Parse results and draw bounding boxes
            for *xyxy, conf, cls in results.xyxy[0]:
                cx, cy = (int(xyxy[0]+(xyxy[2]-xyxy[0])/2) , int(xyxy[1]+(xyxy[3]-xyxy[1])/2))
                        #cv2.line(current_frame_small, (0, h-400), (w, h-400), (0,255,0), thickness=3)
                cv2.line(current_frame_small, (0, 250), (w, 250), (0,255,0), thickness=3)
                       # cv2.line(current_frame_small, (0, 350), (w, 350), (0,0,255), thickness=3)
                #lbl_cy['text']= ": " + str(int(cy))
                lbl_sabes['text']= ": " + str(sabes_count)
                lbl_batu['text']= ": " + str(batu_count)
                #lbl_val['text']= "skp: " + str(skip_double)
                if conf>0.5:
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    lbl_val['text']=  label #"skp: " + str(skip_double)
                    print(label)
                    if imageVal:
                        if(int(cls) == 1) or (int(cls) == 2):
                            if (int(cls) == 1):
                                cv2.rectangle(current_frame_small, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,255), 2)
                                cv2.putText(current_frame_small, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                            if (int(cls) == 2):
                                cv2.rectangle(current_frame_small, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,255,0), 2)
                                cv2.putText(current_frame_small, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

                    if (cy > 250) and (int(cls) == 1) and (skip_double == 0):
                        skip_double = 1
                        sabes_count += 1

                        cntFileSaveSabes +=1

                        current_time = datetime.datetime.now()
                        tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
                        jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)

                        filenameSave = "Img_" + str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) +"_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second) + "_0S.jpg"
                        filename = os.path.join(os.getcwd() + "\\images\\", "res_"+filenameSave)
                        print(filename)


                        img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                        cv2.imwrite(filename, img_resize)
                        cv2.imwrite("tempImg.jpg", current_frame_small)

                        #cv2.imwrite(filename, img_resize)

                        UploadIMG(filenameSave,filename, '0','live CCTV')

                    if (cy > 250) and (int(cls) == 2) and (skip_double == 0):
                        skip_double = 1
                        batu_count += 1

                        cntFileSaveBatu +=1
                        #filename = f'savedImageBatu_{cntFileSaveBatu}.jpg'
                        current_time = datetime.datetime.now()
                        tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
                        jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)
                        filenameSave = "Img_" + str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) +"_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second) + "_1B.jpg"
                        filename = os.path.join(os.getcwd() + "\\images\\", "res_"+filenameSave)
                        print(filename)
                        print(filenameSave)

                        img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                        cv2.imwrite(filename, img_resize)
                        cv2.imwrite("tempImg.jpg", current_frame_small)
                        #cv2.imwrite(filename, img_resize)

                        UploadIMG(filenameSave,filename, '1','live CCTV')
                    if (cy <= 200) and ((int(cls) == 1) or (int(cls) == 2)):
                        skip_double = 0

            #frame3 = cv2.flip(frame, 1)
            if imageVal:
                frame3 = current_frame_small
                cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
                img2 = Image.fromarray(cv2image2)

                imgtk2 = ImageTk.PhotoImage(image=img2)
                lmain1.imgtk = imgtk2
                lmain1.configure(image=imgtk2)
            else:
                isExistTempImg = os.path.exists(os.getcwd()+'\\'+"tempImg.jpg")
                isExistTempImgDetect = os.path.exists(os.getcwd()+'\\'+"tempImgDetect.jpg")
                if isExistTempImgDetect:
                    imageTemp = cv2.imread("tempImgDetect.jpg")
                if isExistTempImg:
                    imageTemp = cv2.imread("tempImg.jpg")
                if isExistTempImg or isExistTempImgDetect:
                    cv2image2 = cv2.cvtColor(imageTemp, cv2.COLOR_BGR2RGBA)
                    img2 = Image.fromarray(cv2image2)
                    imgtk2 = ImageTk.PhotoImage(image=img2)

                    lmain1.imgtk = imgtk2
                    lmain1.configure(image=imgtk2)

            lmain1.after(10, show_frame)
    #while (cap.isOpened):
    show_frame()

def UploadIMG(filenameSave,filename, muatan,source):
    if connect():
        current_time = datetime.datetime.now()
        tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
        jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)

        url = 'https://produk-inovatif.com/latihan/galian/galian.php?ins=2'
        #data = { 'tgl': '2023-10-26', 'jam': '21:31:00', 'muatan': '1', 'pintu': '4', 'filename': 'test.jpg', 'source': 'upload'}
        data = { 'tgl': tgl, 'jam': jam, 'muatan': muatan, 'pintu': str(PINTU), 'filename': filenameSave, 'source': source}

        head = {'Content-Type' : 'application/x-www-form-urlencoded'}
        x = requests.post(url, data=data, headers=head)


        print(x.text)

        #----- update data di tbtempdata
        url = 'https://produk-inovatif.com/latihan/galian/galian.php?muatan=2'
        #data = {  'sabes': '2', 'batubelah': '0', 'pintu': '2'}
        data = {   'tgl': tgl, 'jam': jam,'muatan': muatan, 'pintu': str(PINTU)}

        head = {'Content-Type' : 'application/x-www-form-urlencoded'}
        x = requests.post(url, data=data, headers=head)

        print(x.text)



        dfile = open(filename, "rb").read()

        url_img = 'https://produk-inovatif.com/latihan/galian/img_py.php'
        files= {'file': (filenameSave,dfile,'image/jpg',{'Expires': '0'}) }
        test_res = requests.post(url_img, files =  files)
        print(test_res)
        print(test_res.ok)
        if test_res.ok:
            isExistTempImg = os.path.exists(filename)
            print(isExistTempImg)
            if isExistTempImg:
                os.remove(filename)
                return True

def reUPLOAD_img():
    path_img = os.getcwd()+"\\images"
    #print(path_img)

    imgOFF = os.listdir(path_img)
    cek_img_off = len(imgOFF)
    print(cek_img_off)
    if cek_img_off > 0:
        fileN = imgOFF[0]
        #print(fileN)
        #print(os.path.join(os.getcwd() + "\\images\\", fileN))
        #print(fileN[-6])
        if UploadIMG(fileN,os.path.join(os.getcwd() + "\\images\\", fileN),fileN[-6],'re-upload'):
            print("Upload offline suksess...")

def upload_vid_func():
    print(imageVal)

    def browse_file():
        def run_yolov5_on_video():
            isExistTempImg = os.path.exists(os.getcwd()+'\\'+"tempImg.jpg")
            print(os.getcwd()+'\\'+"tempImg.jpg")
            print(isExistTempImg)
            if isExistTempImg:
                os.remove("tempImg.jpg")
            isExistTempImgDetect = os.path.exists(os.getcwd()+'\\'+"tempImgDetect.jpg")
            if isExistTempImgDetect:
                os.remove("tempImgDetect.jpg")

            width, height = 1200, 1200
            cap = cv2.VideoCapture(file_path)

            display_frame2 = tk.Frame(root)
            display_frame2.place(relx=0.5, rely=0.3, width = 800, height = 900, anchor=tk.CENTER)


            lmain1 = tk.Label(display_frame2)
            lmain1.place(x = 0, y = 0, width=800, height=900)

            def show_frame():
                    global skip_double,sabes_count, batu_count, cntFileSaveBatu, cntFileSaveSabes, cntFlag
                    _, frame = cap.read()
                    if frame is None:
                        return
                    else:
                        w, h = frame.shape[1],frame.shape[0]

                    #print(str(w) + " h: " + str(h))

                    current_frame_small = cv2.resize(frame,(0,0),fx=0.35,fy=0.35)
                    # Perform inference
                    results = model(current_frame_small)
                    print(".")
                    lbl_val['text']=  "...."

                    reUPLOAD_img()

                    if cntFlag==0:
                        cv2.imwrite("tempImgDetect.jpg", current_frame_small)
                        cntFlag = 1



                    # Parse results and draw bounding boxes
                    for *xyxy, conf, cls in results.xyxy[0]:
                        cx, cy = (int(xyxy[0]+(xyxy[2]-xyxy[0])/2) , int(xyxy[1]+(xyxy[3]-xyxy[1])/2))
                        if imageVal:
                            cv2.line(current_frame_small, (0, 250), (w, 250), (0,255,0), thickness=3)
                       # cv2.line(current_frame_small, (0, 350), (w, 350), (0,0,255), thickness=3)
                        #lbl_cy['text']= ": " + str(int(cy))
                        lbl_sabes['text']= ": " + str(sabes_count)
                        lbl_batu['text']= ": " + str(batu_count)

                        label = f'{model.names[int(cls)]} {conf:.2f}'
                        lbl_val['text']=  label #"skp: " + str(skip_double)
                        print(label)


                        #cv2.imwrite("tempImgDetect.jpg", current_frame_small)
                        if conf>0.5:

                            if imageVal:
                                if(int(cls) == 1) or (int(cls) == 2):
                                    if (int(cls) == 1):
                                        cv2.rectangle(current_frame_small, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,255), 2)
                                        cv2.putText(current_frame_small, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                                    if (int(cls) == 2):
                                        cv2.rectangle(current_frame_small, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,255,0), 2)
                                        cv2.putText(current_frame_small, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

                            if (cy > 250) and (int(cls) == 1) and (skip_double == 0):
                                skip_double = 1
                                sabes_count += 1

                                cntFileSaveSabes +=1
                                
                                current_time = datetime.datetime.now()
                                tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
                                jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)

                                filenameSave = "Img_" + str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) +"_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second) + "_0S.jpg"
                                filename = os.path.join(os.getcwd() + "\\images\\", "res_"+filenameSave)
                                print(filename)


                                img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                                cv2.imwrite(filename, img_resize)
                                cv2.imwrite("tempImg.jpg", current_frame_small)

                                #cv2.imwrite(filename, img_resize)

                                UploadIMG(filenameSave,filename, '0','recorded')

                            if (cy > 250) and (int(cls) == 2) and (skip_double == 0):
                                skip_double = 1
                                batu_count += 1

                                cntFileSaveBatu +=1
                                #filename = f'savedImageBatu_{cntFileSaveBatu}.jpg'
                                current_time = datetime.datetime.now()
                                tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
                                jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)
                                filenameSave = "Img_" + str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) +"_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second) + "_1B.jpg"
                                filename = os.path.join(os.getcwd() + "\\images\\", "res_"+filenameSave)
                                print(filename)
                                print(filenameSave)

                                img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                                cv2.imwrite(filename, img_resize)
                                cv2.imwrite("tempImg.jpg", current_frame_small)
                                #cv2.imwrite(filename, img_resize)

                                UploadIMG(filenameSave,filename, '1','recorded')

                            if (cy <= 200) and ((int(cls) == 1) or (int(cls) == 2)):
                                skip_double = 0

                    #frame3 = cv2.flip(frame, 1)
                    if imageVal:
                        frame3 = current_frame_small
                        cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
                        img2 = Image.fromarray(cv2image2)

                        imgtk2 = ImageTk.PhotoImage(image=img2)
                        lmain1.imgtk = imgtk2
                        lmain1.configure(image=imgtk2)
                    else:
                        isExistTempImg = os.path.exists(os.getcwd()+'\\'+"tempImg.jpg")
                        isExistTempImgDetect = os.path.exists(os.getcwd()+'\\'+"tempImgDetect.jpg")
                        if isExistTempImgDetect:
                            imageTemp = cv2.imread("tempImgDetect.jpg")
                        if isExistTempImg:
                            imageTemp = cv2.imread("tempImg.jpg")
                        if isExistTempImg or isExistTempImgDetect:
                            cv2image2 = cv2.cvtColor(imageTemp, cv2.COLOR_BGR2RGBA)
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

    lbl_val['text']=  "DONE..."





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

btn_rdcsv = Button(fr_button, text="Webcam", command=web_cam_func)
btn_wrcsv = Button(fr_button, text="Upload Video", command=upload_vid_func)
btn_baseline = Button(fr_button, text="CCTV",command=cctv_func)
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

lbl_phase_min = Label(fr_button, text="Pintu", font=('Times 14'))
lbl_phase_min.grid(row=18, column=0, padx=10, pady=5, sticky="w")
lbl_cy = Label(fr_button, text=": 0", font=('Times 14'))
lbl_cy.grid(row=18, column=1, padx=5, pady=5, sticky="w")

lbl_phase_max = Label(fr_button, text="Object: None", font=('Times 14'))
#lbl_phase_max.grid(row=19, column=0, padx=10, pady=5, sticky="w")
lbl_val = Label(fr_button, text="Object: None", font=('Times 14'))
lbl_val.grid(row=19, column=0, padx=5, pady=5, sticky="w")

lbl_rtsp1 = Label(fr_button, text="RTSP", font=('Times 14'))
#lbl_rtsp1.grid(row=20, column=0, padx=10, pady=5, sticky="w")
lbl_rtsp1val = Label(fr_button, text=": 0", font=('Times 14'))
#lbl_rtsp1val.grid(row=20, column=1, padx=5, pady=5, sticky="w")

# textbox
lbl_rtsp = Label(fr_button, text="RTSP: ", font=('Times 14'))
#lbl_rtsp.grid(row=25, column=0, padx=10, pady=5, sticky="w")
input_text = StringVar() 
  
entry1 = Entry(fr_button, width=25, textvariable = input_text, justify = CENTER)
entry1.focus_force() 
#entry1.pack(side = TOP, ipadx = 30, ipady = 6)
entry1.grid(row=26, column=0, padx=5, pady=5, sticky="w")
#-------------------------------------
result_label1 = Label(fr_result, text="Crack Estimation:                            ", font=("Bold"))
#================== end GUI ==========
fr_button.grid(row=0, column=0, sticky="ns")
fr_graph.grid(row=0, column=1, sticky="nsew")
fr_result.grid(row=0, column=2, sticky="nsew")



entry1.insert(0, val_rtsp)
lbl_cy['text']= ": " + str(PINTU)
lbl_rtsp1val['text']= ": " + val_rtsp
#get_location()
cctv_func()

root.mainloop()