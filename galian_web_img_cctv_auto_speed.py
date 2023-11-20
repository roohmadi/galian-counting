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
import time
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

global skip_double,sabes_count, batu_count, cntFileSaveSabes, cntFileSaveBatu, PINTU, img_del_date, host, weight_file, OStype
global cntdot
cntdot = 0
skip_double = 0
sabes_count =  0
batu_count = 0
cntFileSaveSabes = 0
cntFileSaveBatu = 0
cntFlag = 0




from configparser import ConfigParser
config = ConfigParser()
isExistINI = os.path.exists('galianset.ini')
print(isExistINI)
if isExistINI:
    config.read('galianset.ini')
    val_rtsp = config.get('galian', 'rtspset')
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
print("host: " + host)
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
print("cctv: " + val_rtsp)
isExistupl = os.path.exists(path_imgupl)
if isExistupl:
    print("folder upl exist")
else:
    print("folder upl not exist")
    os.mkdir(path_imgupl)
isExistimg = os.path.exists(path_img)
if isExistimg:
    print("folder images exist")
else:
    print("folder images not exist")
    os.mkdir(path_img)

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
    def __init__(self, window, window_title, video_source=val_rtsp):

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
        self.btn_rdcsv = Button(fr_button, text="Webcam", )
        self.btn_wrcsv = Button(fr_button, text="Upload Video",command = self.upload_vid_func)
        self.btn_baseline = Button(fr_button, text="Open CCTV",command = self.cctv_func)
        self.btn_quit = Button(fr_button, text="Quit", command=self.window.destroy)

        self.btn_rdcsv.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.btn_wrcsv.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        self.btn_baseline.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        self.btn_quit.grid(row=3, column=0, sticky="ew", padx=10, pady=10)

        self.btn_rdcsv.config(width=10, height=5)
        self.btn_wrcsv.config(width=10, height=5)
        self.btn_baseline.config(width=10, height=5)
        self.btn_quit.config(width=10, height=5)
        self.btn_rdcsv["state"] = DISABLED

        self.lbl_obj = Label(fr_button, text="Summary:", font=('Times 14'))
        self.lbl_obj.grid(row=15, column=0, padx=10, pady=5, sticky="w")

        self.lbl_mag_min = Label(fr_button, text="Sabes", font=('Times 14'))
        self.lbl_mag_min.grid(row=16, column=0, padx=10, pady=5, sticky="w")
        self.lbl_sabes = Label(fr_button, text=": 0 ", font=('Times 14'))
        self.lbl_sabes.grid(row=16, column=1, padx=5, pady=5, sticky="w")

        self.lbl_mag_max = Label(fr_button, text="Batu Belah", font=('Times 14'))
        self.lbl_mag_max.grid(row=17, column=0, padx=10, pady=5, sticky="w")
        self.lbl_batu = Label(fr_button, text=": 0 ", font=('Times 14'))
        self.lbl_batu.grid(row=17, column=1, padx=5, pady=5, sticky="w")

        self.lbl_phase_min = Label(fr_button, text="Pintu", font=('Times 14'))
        self.lbl_phase_min.grid(row=18, column=0, padx=10, pady=5, sticky="w")
        self.lbl_cy = Label(fr_button, text=": 0", font=('Times 14'))
        self.lbl_cy.grid(row=18, column=1, padx=5, pady=5, sticky="w")

        self.lbl_phase_max = Label(fr_button, text="Object ", font=('Times 14'))
        self.lbl_phase_max.grid(row=19, column=0, padx=10, pady=5, sticky="w")
        self.lbl_val = Label(fr_button, text=": None", font=('Times 14'))
        self.lbl_val.grid(row=19, column=1, padx=5, pady=5, sticky="w")

        self.lbl_rtsp1 = Label(fr_button, text="Status: ", font=('Times 14'))
        self.lbl_rtsp1.grid(row=20, column=0, padx=10, pady=5, sticky="w")
        self.lbl_rtsp1val = Label(fr_button, text=": None", font=('Times 14'))
        self.lbl_rtsp1val.grid(row=20, column=1, padx=5, pady=5, sticky="w")

        # textbox
        self.lbl_rtsp = Label(fr_button, text="RTSP: ", font=('Times 14'))
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

        entry1.insert(0, val_rtsp)
        self.lbl_cy['text']= ": " + str(PINTU)
        #self.lbl_rtsp1val['text']= ": " + val_rtsp

        self.window.mainloop()

    def connect(self,host='http://google.com'):
        try:
            urllib.request.urlopen(host) #Python 3.x
            return True
        except:
            return False

    def cctv_close(self):
        if self.vid.isOpened():
            self.vid.Close()
            self.vid.release()

    def cctv_func(self):
        self.btn_rdcsv["state"]=NORMAL
        self.video_source = val_rtsp

        isExistTempImg = os.path.exists(path_imgTemp)
        print(path_imgTemp)
        print(isExistTempImg)
        if isExistTempImg:
            os.remove("tempImg.jpg")
        isExistTempImgDetect = os.path.exists(path_imgTempDetect)
        if isExistTempImgDetect:
            os.remove("tempImgDetect.jpg")

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        
        
        #self.vid = MyVideoCapture('E:\YOLO\counting\vechicle-counting-yolo-main\data\video\1013.mp4')

        display_frame2 = tkinter.Frame(self.window)
        display_frame2.place(relx=0.6, rely=0.3, width = 800, height = 900, anchor=tkinter.CENTER)
        self.lmain1 = tkinter.Label(display_frame2)
        self.lmain1.place(x = 0, y = 0, width=800, height=900)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 10
        self.update()

    def reUPLOAD_img(self):
        #print(path_img)

        imgOFF = os.listdir(path_img)
        cek_img_off = len(imgOFF)
        #print(cek_img_off)
        if cek_img_off > 0:
            fileN = imgOFF[0]
            #print(fileN)
            #print(os.path.join(os.getcwd() + "\\images\\", fileN))
            #print(fileN[-6])
            if OSWindows:
                if self.UploadIMG(fileN,os.path.join(os.getcwd() + "\\images\\", fileN),fileN[-6],'re-upload'):
                    print("Upload offline suksess...")
            else:
                if self.UploadIMG(fileN,os.path.join(os.getcwd() + "/images/", fileN),fileN[-6],'re-upload'):
                    print("Upload offline suksess...")

    def delete_old_img(self):
        for fileN in os.listdir(path_imgupl):
            x = fileN.split("_")
            #print(x)
            imgUPL = x[0]
            #print(imgUPL)
            cek_img_del = len(imgUPL)
            #print(cek_img_del)
            if (cek_img_del > 0) and (imgUPL == 'upl'):
                today = datetime.date.today()
                year = today.year
                if (x[2] == str(year)):
                    date_create = date(int(x[2]),int(x[3]),int(x[4]))
                    get_diff_days = (date.today() - date_create).days
                elif (x[3] == str(year)):
                    date_create = date(int(x[3]),int(x[4]),int(x[5]))
                    get_diff_days = (date.today() - date_create).days
                elif (x[4] == str(year)):
                    date_create = date(int(x[4]),int(x[5]),int(x[6]))
                    get_diff_days = (date.today() - date_create).days


                if get_diff_days > img_del_date:
                    #print(path_img + fileN)
                    isExistdelUPL = os.path.exists(path_imgupl + fileN)
                    #print(isExistdelUPL)
                    if isExistdelUPL:
                        os.remove(path_imgupl + fileN)
                        isExistdelUPL = os.path.exists(path_imgupl + fileN)
                        if isExistdelUPL:
                            print("file " + fileN + " gagal dihapus")
                        else:
                            print("file " + fileN + " telah dihapus")
                                #else:
                                    #    print("NONE")

    def UploadIMG(self,filenameSave, filename, muatan, source):
        if self.connect():
            current_time = datetime.datetime.now()
            tgl = str(current_time.year) + "-" + \
                str(current_time.month) + "-" + str(current_time.day)
            jam = str(current_time.hour) + ":" + \
                str(current_time.minute) + ":" + str(current_time.second)

            #url = 'https://produk-inovatif.com/latihan/galian/galian.php?ins=2'
            #url = 'https://produk-inovatif.com/latihan/galian/galian.php?ins=2'
            url = host + '/galian.php?ins=2'
            #print (url)
            #data = { 'tgl': '2023-10-26', 'jam': '21:31:00', 'muatan': '1', 'pintu': '4', 'filename': 'test.jpg', 'source': 'upload'}
            data = {'tgl': tgl, 'jam': jam, 'muatan': muatan, 'pintu': str(
                PINTU), 'filename': filenameSave, 'source': source}

            head = {'Content-Type': 'application/x-www-form-urlencoded'}
            x = requests.post(url, data=data, headers=head)

            print(x.text)

            #----- update data di tbtempdata
            #url = 'https://produk-inovatif.com/latihan/galian/galian.php?muatan=2'
            #url = 'https://produk-inovatif.com/latihan/galian/galian.php?muatan=2'
            url = host + '/galian.php?muatan=2'
            #print (url)
            #data = {  'sabes': '2', 'batubelah': '0', 'pintu': '2'}
            data = {'tgl': tgl, 'jam': jam, 'muatan': muatan, 'pintu': str(PINTU)}

            head = {'Content-Type': 'application/x-www-form-urlencoded'}
            x = requests.post(url, data=data, headers=head)

            #print(x.text)

            dfile = open(filename, "rb").read()

            #url_img = 'https://produk-inovatif.com/latihan/galian/img_py.php'
            #url_img = 'https://produk-inovatif.com/latihan/galian/img_py.php'
            url_img = host + '/img_py.php'
            #print (url_img)
            files = {'file': (filenameSave, dfile, 'image/jpg', {'Expires': '0'})}
            test_res = requests.post(url_img, files=files)
            #print(test_res)
            #print(test_res.ok)
            if test_res.ok:
                isExistTempImg = os.path.exists(filename)
                #print(isExistTempImg)
                if isExistTempImg:
                    #img_del_date
                    #os.remove(filename)
                    #print("belum rename: ")
                    #print(filename)
                    x = filename.split("_")
                    xxpath = x[0]
                    panj = len(xxpath)
                    imgUPL = xxpath[panj-3:panj]
                    #print(xxpath[panj-3:panj])
                    #print(imgUPL)
                    if (imgUPL == 'upl'):
                        print("file upl sudah ada")
                    else:
                        newfilename = path_img + "upl_" + filenameSave
                        newfileloc = path_imgupl + "upl_" + filenameSave
                        #print("=====>")
                        #print(filename)
                        #print(newfilename)
                        #print(newfileloc)
                        #print(filenameSave)
                        os.rename(filename, newfilename)
                        shutil.move(newfilename,newfileloc)
                        print("sudah rename: ")

                    return True
    
    def update_img_temp (self,frameTemp):
        #cv2image2 = cv2.cvtColor(frameTemp, cv2.COLOR_BGR2RGBA)
        #img2 = Image.fromarray(cv2image2)
        #imgtk2 = ImageTk.PhotoImage(image=img2)
        
        #self.lmain1.imgtk = imgtk2
        #self.lmain1.configure(image=imgtk2)
        
        frame3 = frameTemp
        cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
        img2 = Image.fromarray(cv2image2)
        
        imgtk2 = ImageTk.PhotoImage(image=img2)
        self.lmain1.imgtk = imgtk2
        self.lmain1.configure(image=imgtk2)
        
        
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
                print(file_path)
                self.fps = cap.get(cv2.CAP_PROP_FPS)
                print("fps: " + str(self.fps))
                print(file_path)

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
                    current_frame_small = cv2.resize(frame,(0,0),fx=0.8,fy=0.8)
                    
                    if cntdot > self.fps:
                        cntdot = 0
                    if cntdot == (self.fps/2):
                        results = model(current_frame_small)
                        
                        self.lbl_val['text'] = ": None"
                        
                        
                        self.reUPLOAD_img()
                        self.delete_old_img()
                        
                        if cntFlag == 0:
                            cv2.imwrite("tempImgDetect.jpg", current_frame_small)
                            cntFlag = 1
                        for *xyxy, conf, cls in results.xyxy[0]:
                            cx, cy = (int(xyxy[0]+(xyxy[2]-xyxy[0])/2),
                                  int(xyxy[1]+(xyxy[3]-xyxy[1])/2))
                            cv2.line(current_frame_small, (0, 250),
                                 (w, 250), (0, 255, 0), thickness=3)
                            self.lbl_sabes['text'] = ": " + str(sabes_count)
                            self.lbl_batu['text'] = ": " + str(batu_count)
                            label = f'{model.names[int(cls)]} {conf:.2f}'
                            print(label)
                            if conf>0.5:
                                #label = f'{model.names[int(cls)]} {conf:.2f}'
                                label = f'{model.names[int(cls)]}'
                                self.lbl_val['text']=  ": " + label
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
                                    if OSWindows:
                                        filename = os.path.join(os.getcwd() + "\\images\\", "res_"+filenameSave)
                                    else:
                                        filename = os.path.join(os.getcwd() + "/images/", "res_"+filenameSave)

                                    print(filename)


                                    img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                                    cv2.imwrite(filename, img_resize)
                                    cv2.imwrite("tempImg.jpg", current_frame_small)

                                    #cv2.imwrite(filename, img_resize)

                                    self.UploadIMG( filenameSave,filename, '0','recorded')

                                if (cy > 250) and (int(cls) == 2) and (skip_double == 0):
                                    skip_double = 1
                                    batu_count += 1

                                    cntFileSaveBatu +=1
                                    #filename = f'savedImageBatu_{cntFileSaveBatu}.jpg'
                                    current_time = datetime.datetime.now()
                                    tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
                                    jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)
                                    filenameSave = "Img_" + str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) +"_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second) + "_1B.jpg"
                                    if OSWindows:
                                        filename = os.path.join(os.getcwd() + "\\images\\", "res_"+filenameSave)
                                    else:
                                        filename = os.path.join(os.getcwd() + "/images/", "res_"+filenameSave)
                                    print(filename)
                                    print(filenameSave)

                                    img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                                    cv2.imwrite(filename, img_resize)
                                    cv2.imwrite("tempImg.jpg", current_frame_small)
                                    #cv2.imwrite(filename, img_resize)

                                    self.UploadIMG(filenameSave,filename, '1','recorded')
                                if (cy <= 200) and ((int(cls) == 1) or (int(cls) == 2)):
                                    skip_double = 0
                        
                        #-----
                        cv2image2 = cv2.cvtColor(current_frame_small, cv2.COLOR_BGR2RGBA)
                        img2 = Image.fromarray(cv2image2)
                        imgtk2 = ImageTk.PhotoImage(image=img2)

                        self.lmain1.imgtk = imgtk2
                        self.lmain1.configure(image=imgtk2)
                        
                        
                    cntdot +=1
                    self.lbl_rtsp1val['text'] = ": " + str(cntdot)
                    #print(cntdot)
                    
                    
                    
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

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def update(self):
        global skip_double,sabes_count, batu_count, cntFileSaveBatu, cntFileSaveSabes, cntFlag,cntdot
        fps,ret, frame = self.vid.get_frame()
        print("fps: " + str(fps))
        if ret:
            if frame is None:
                return
            else:
                w, h = frame.shape[1],frame.shape[0]
            current_frame_small = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
                    
            if cntdot > fps:
                cntdot = 0
            if cntdot == (fps/2):
                results = model(current_frame_small)
                self.lbl_val['text'] = ": None"                        
                        
                self.reUPLOAD_img()
                self.delete_old_img()
                        
                if cntFlag == 0:
                    cv2.imwrite("tempImgDetect.jpg", current_frame_small)
                    cntFlag = 1
                for *xyxy, conf, cls in results.xyxy[0]:
                    cx, cy = (int(xyxy[0]+(xyxy[2]-xyxy[0])/2),
                                  int(xyxy[1]+(xyxy[3]-xyxy[1])/2))
                    cv2.line(current_frame_small, (0, 250),
                                 (w, 250), (0, 255, 0), thickness=3)
                    self.lbl_sabes['text'] = ": " + str(sabes_count)
                    self.lbl_batu['text'] = ": " + str(batu_count)
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    print(label)
                    if conf>0.5:
                        #label = f'{model.names[int(cls)]} {conf:.2f}'
                        label = f'{model.names[int(cls)]}'
                        self.lbl_val['text']=  ": " + label
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
                            if OSWindows:
                                filename = os.path.join(os.getcwd() + "\\images\\", "res_"+filenameSave)
                            else:
                                filename = os.path.join(os.getcwd() + "/images/", "res_"+filenameSave)

                            print(filename)


                            img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                            cv2.imwrite(filename, img_resize)
                            cv2.imwrite("tempImg.jpg", current_frame_small)

                            #cv2.imwrite(filename, img_resize)

                            self.UploadIMG( filenameSave,filename, '0','recorded')

                        if (cy > 250) and (int(cls) == 2) and (skip_double == 0):
                            skip_double = 1
                            batu_count += 1

                            cntFileSaveBatu +=1
                            #filename = f'savedImageBatu_{cntFileSaveBatu}.jpg'
                            current_time = datetime.datetime.now()
                            tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
                            jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)
                            filenameSave = "Img_" + str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) +"_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second) + "_1B.jpg"
                            if OSWindows:
                                filename = os.path.join(os.getcwd() + "\\images\\", "res_"+filenameSave)
                            else:
                                filename = os.path.join(os.getcwd() + "/images/", "res_"+filenameSave)
                            print(filename)
                            print(filenameSave)
                            
                            img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                            cv2.imwrite(filename, img_resize)
                            cv2.imwrite("tempImg.jpg", current_frame_small)
                            #cv2.imwrite(filename, img_resize)

                            self.UploadIMG(filenameSave,filename, '1','recorded')
                        if (cy <= 200) and ((int(cls) == 1) or (int(cls) == 2)):
                            skip_double = 0
                
                cv2image2 = cv2.cvtColor(current_frame_small, cv2.COLOR_BGR2RGBA)
                img2 = Image.fromarray(cv2image2)
                imgtk2 = ImageTk.PhotoImage(image=img2)

                self.lmain1.imgtk = imgtk2
                self.lmain1.configure(image=imgtk2)
                        
                        
            cntdot +=1
            self.lbl_rtsp1val['text'] = ": " + str(cntdot)
            
        self.window.after(self.delay, self.update)
        
    def update1(self):
        global skip_double,sabes_count, batu_count, cntFileSaveBatu, cntFileSaveSabes, cntFlag,cntdot
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            if frame is None:
                return
            else:
                w, h = frame.shape[1],frame.shape[0]

            current_frame_small = cv2.resize(frame,(0,0),fx=0.8,fy=0.8)
            results = model(current_frame_small)

            if cntdot > 9:
                #print(".")
                cntdot = 0
                #self.update_img_temp (current_frame_small)
                isExistTempImg = os.path.exists(path_imgTemp)
                #print(path_imgTemp)
                #print(isExistTempImg)
                #print(isExistTempImg)
                if isExistTempImg:
                    os.remove(path_imgTemp)
                    time.sleep(2)
                    isExistTempImg = os.path.exists(path_imgTemp)
                    #print(isExistTempImg)
                    cv2.imwrite("tempImg.jpg", current_frame_small)
                    cv2image2 = cv2.cvtColor(current_frame_small, cv2.COLOR_BGR2RGBA)
                    img2 = Image.fromarray(cv2image2)
                    imgtk2 = ImageTk.PhotoImage(image=img2)

                    self.lmain1.imgtk = imgtk2
                    self.lmain1.configure(image=imgtk2)
                
            cntdot += 1
            #print("-")
            print(cntdot)
            self.lbl_val['text'] = ": None"

            self.lbl_rtsp1val['text'] = ": " + str(cntdot)
            self.reUPLOAD_img()
            self.delete_old_img()

            if cntFlag == 0:
                cv2.imwrite("tempImgDetect.jpg", current_frame_small)
                cntFlag = 1

            # Parse results and draw bounding boxes
            for *xyxy, conf, cls in results.xyxy[0]:
                cx, cy = (int(xyxy[0]+(xyxy[2]-xyxy[0])/2),int(xyxy[1]+(xyxy[3]-xyxy[1])/2))
                cv2.line(current_frame_small, (0, 250),
                         (w, 250), (0, 255, 0), thickness=3)
                self.lbl_sabes['text'] = ": " + str(sabes_count)
                self.lbl_batu['text'] = ": " + str(batu_count)

                if conf>0.5:
                    #label = f'{model.names[int(cls)]} {conf:.2f}'
                    label = f'{model.names[int(cls)]}'
                    self.lbl_val['text']=  ": " + label #"skp: " + str(skip_double)
                    #print(label)
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
                        if OSWindows:
                            filename = os.path.join(os.getcwd() + "\\images\\", "res_"+filenameSave)
                        else:
                            filename = os.path.join(os.getcwd() + "/images/", "res_"+filenameSave)

                        print(filename)


                        img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                        cv2.imwrite(filename, img_resize)
                        cv2.imwrite("tempImg.jpg", current_frame_small)

                        #cv2.imwrite(filename, img_resize)

                        self.UploadIMG( filenameSave,filename, '0','live CCTV')

                    if (cy > 250) and (int(cls) == 2) and (skip_double == 0):
                        skip_double = 1
                        batu_count += 1

                        cntFileSaveBatu +=1
                        #filename = f'savedImageBatu_{cntFileSaveBatu}.jpg'
                        current_time = datetime.datetime.now()
                        tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
                        jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)
                        filenameSave = "Img_" + str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) +"_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second) + "_1B.jpg"
                        if OSWindows:
                            filename = os.path.join(os.getcwd() + "\\images\\", "res_"+filenameSave)
                        else:
                            filename = os.path.join(os.getcwd() + "/images/", "res_"+filenameSave)
                        print(filename)
                        print(filenameSave)

                        img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                        cv2.imwrite(filename, img_resize)
                        cv2.imwrite("tempImg.jpg", current_frame_small)
                        #cv2.imwrite(filename, img_resize)

                        self.UploadIMG(filenameSave,filename, '1','live CCTV')
                    if (cy <= 200) and ((int(cls) == 1) or (int(cls) == 2)):
                        skip_double = 0
                        
                

            #-----

            #frame3 = current_frame_small
            #cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
            #img2 = Image.fromarray(cv2image2)
            #imgtk2 = ImageTk.PhotoImage(image=img2)
            #self.lmain1.imgtk = imgtk2
            #self.lmain1.configure(image=imgtk2)
            #self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            #self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            #print(imageVal)
            if imageVal:
                frame3 = current_frame_small
                cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
                img2 = Image.fromarray(cv2image2)

                imgtk2 = ImageTk.PhotoImage(image=img2)
                self.lmain1.imgtk = imgtk2
                self.lmain1.configure(image=imgtk2)
            else:
                isExistTempImg = os.path.exists(path_imgTemp)
                isExistTempImgDetect = os.path.exists(path_imgTempDetect)
                #print(isExistTempImg)
                #print(isExistTempImgDetect)
                if isExistTempImgDetect:
                    imageTemp = cv2.imread("tempImgDetect.jpg")
                if isExistTempImg:
                    imageTemp = cv2.imread("tempImg.jpg")
                if isExistTempImg or isExistTempImgDetect:
                    cv2image2 = cv2.cvtColor(imageTemp, cv2.COLOR_BGR2RGBA)
                    img2 = Image.fromarray(cv2image2)
                    imgtk2 = ImageTk.PhotoImage(image=img2)

                    self.lmain1.imgtk = imgtk2
                    self.lmain1.configure(image=imgtk2)

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
            fps = self.vid.get(cv2.CAP_PROP_FPS)
            
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (fps,ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(tkinter.Tk(), "GALIAN C COUNTER. 1.1 --fps")