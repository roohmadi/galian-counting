# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:29:54 2023

@author: roohm
"""

# -*- coding: utf-8 -*-

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

global  YlineDetect0,YlineDetect1
YlineDetect1 = 230
YlineDetect0 = 100

# importing "cmath" for complex number operations
import cmath
from configparser import ConfigParser


import warnings
warnings.filterwarnings("ignore")
running = False
imageVal = True
saveTempImgFlag = False

production = False

global skip_double,sabes_count, batu_count, cntFileSaveSabes, cntFileSaveBatu, PINTU, img_del_date, host, weight_file, OStype
global cntdot, tempCy, chtruk, skip_double0, chtruk0
chtruk0 = 0
chtruk = 0
cntdot = 0
tempCy = 0
skip_double0 = 0
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
        self.btn_baseline = Button(fr_button, text="Open CCTV",)
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
        self.btn_baseline["state"] = DISABLED

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

        fr_button.grid(row=0, column=0, sticky="ns")
        fr_graph.grid(row=0, column=1, sticky="nsew")
        fr_result.grid(row=0, column=2, sticky="nsew")

       # entry1.insert(0, val_rtsp)
        self.lbl_cy['text']= ": " + str(PINTU)
        #self.lbl_rtsp1val['text']= ": " + val_rtsp

        self.window.mainloop()

    def connect(self,host='http://google.com'):
        try:
            urllib.request.urlopen(host) #Python 3.x
            return True
        except:
            return False


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
                if self.UploadIMGtoPedati(fileN,os.path.join(os.getcwd() + "\\images\\", fileN),fileN[-6],'re-upload'):
                    print("Upload offline suksess...")
            else:
                if self.UploadIMGtoPedati(fileN,os.path.join(os.getcwd() + "/images/", fileN),fileN[-6],'re-upload'):
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
                        #os.remove(path_imgupl + fileN)
                        isExistdelUPL = os.path.exists(path_imgupl + fileN)
                        if isExistdelUPL:
                            print("file " + fileN + " gagal dihapus")
                        else:
                            print("file " + fileN + " telah dihapus")
                                #else:
                                    #    print("NONE")
    def UploadIMGtoPedati(self,filenameSave, filename,filegambarstart,filenamestart, muatan, source):
        from datetime import datetime
        if self.connect():
            hostpedati = 'http://pedati.id:54100/mblb/api/main/insertcapturedouble'
            now = datetime.now()
            print("now =", now)
            date_format = now.strftime("%Y-%m-%d %H:%M:%S")

            #date_obj = datetime.strptime(date_str, date_format)
            print(date_format)
            now = datetime.now()
            tgl_jam = now.strftime("%Y-%m-%d %H:%M:%S")
            dfile = open(filename, "rb").read()
            dfile1 = open(filenamestart, "rb").read()
            #files = {'filegambar': (filenameSave, dfile, 'image/jpg', {'Expires': '0'})}
            files = {'filegambar': (filenameSave, dfile, 'image/jpg', {'Expires': '0'}),'filegambarstart': (filegambarstart, dfile1, 'image/jpg', {'Expires': '0'})}

            #data = {'id_muatan': muatan, 'pintu': str(PINTU), 'tanggal_capture': tgl_jam, 'filename': filenameSave, 'filegambar': (filenameSave, dfile, 'image/jpg', {'Expires': '0'})}
            #data = {'id_muatan': muatan, 'pintu': PINTU, 'tanggal_capture': tgl_jam, 'filename': filenameSave}
            #data = {'id_muatan': filenameSave[-6], 'pintu': PINTU, 'tanggal_capture': tgl_jam, 'filename': filenameSave, 'filegambar': filenameSave}
            data = {'id_muatan': filenameSave[-6], 'pintu': PINTU, 'tanggal_capture': tgl_jam, 'filename': filenameSave, 'filegambar': filenameSave,'filenamestart': filegambarstart, 'filegambarstart': filegambarstart}
    
            head = {'Content-Type': 'application/form-data'}
            print(data)

            test_res = requests.post(hostpedati, data=data,files=files)
			#test_res = requests.post(hostpedati, data=data, files=files)
            #test_res = requests.post(host, data=data)
            print(test_res)
            print(test_res.text)
            if test_res.ok:
                isExistTempImg = os.path.exists(filename)
                if isExistTempImg:
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
                        newfilenamePre = path_img + "upl_" + "preCap_" + filenameSave
                        newfileloc = path_imgupl + "upl_" + filenameSave
                        newfilelocPre = path_imgupl + "upl_" + "preCap_" + filenameSave

                        os.rename(filename, newfilename)
                        os.rename(filenamestart, newfilenamePre)
                        shutil.move(newfilename,newfileloc)
                        shutil.move( newfilenamePre,newfilelocPre)
                        print("sudah rename: ")

                    return True


    
    def update_img_temp (self,frameTemp):
        frame3 = frameTemp
        cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
        img2 = Image.fromarray(cv2image2)
        
        imgtk2 = ImageTk.PhotoImage(image=img2)
        self.lmain1.imgtk = imgtk2
        self.lmain1.configure(image=imgtk2)

    def get_date_time (self):
        current_time = datetime.datetime.now()
        tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
        jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)
        str_date_time = str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) +"_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second)
        return str_date_time

    def detect_muatan(self, results, current_frame_small,w):
        global skip_double,sabes_count, batu_count, cntFileSaveBatu, cntFileSaveSabes, cntFlag
        global chtruk, tempCy, arah
        global skip_double0, chtruk0
        #print("detect muatan")
        for *xyxy, conf, cls in results.xyxy[0]:
            cx, cy = (int(xyxy[0]+(xyxy[2]-xyxy[0])/2),
                  int(xyxy[1]+(xyxy[3]-xyxy[1])/2))
            print("cy: " + str(cy) + "   cyTemp: " + str(tempCy))

            self.lbl_sabes['text'] = ": " + str(sabes_count)
            self.lbl_batu['text'] = ": " + str(batu_count)
            label = f'{model.names[int(cls)]} {conf:.2f} {cls}'
            print(label)
            if conf>0.5:
                #label = f'{model.names[int(cls)]} {conf:.2f}'
                label = f'{model.names[int(cls)]}'
                self.lbl_val['text']=  ": " + label
                if imageVal:
                    if (cy < tempCy):
                        arah = 0
                        tempCy = cy
                        # RED
                        cv2.line(current_frame_small, (0, YlineDetect1),(w, YlineDetect1), (0,0, 255), thickness=3)
                    elif (cy > tempCy) and (chtruk == 1):
                        tempCy = cy
                        arah = 1
                        # GREEN
                        cv2.line(current_frame_small, (0, YlineDetect1),(w, YlineDetect1), (0,255, 0), thickness=3)
                    #--- chek jika ada truk lewat
                    if (cy > (YlineDetect0-100)) and (int(cls) == 0) and (skip_double0 == 0):
                        chtruk0  = 1
                    if (cy > (YlineDetect1-100)) and (int(cls) == 0) and (skip_double == 0):
                        chtruk  = 1
                    #---save deteksi pre muatan
                    if((int(cls) == 1) or (int(cls) == 2)) and (arah == 1) and (chtruk0 == 1) and (skip_double0 == 0):
                        skip_double0 = 1
                        #str_date_time = self.get_date_time()
                        #self.filegambarstart = "preCap_Img_" + str_date_time + ".jpg"
                        #if OSWindows:
                        #    self.filenamestart = os.path.join(os.getcwd() + "\\images\\", "res_"+self.filegambarstart)
                        #else:
                        #    self.filenamestart = os.path.join(os.getcwd() + "/images/", "res_"+self.filegambarstart)

                        #print(self.filenamestart)
                        self.img_resizePrecap = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                        #cv2.imwrite(self.filenamestart, img_resizePrecap)

                        #self.UploadIMGtoPedati(filenameSave, filename,filegambarstart,filenamestart, cls, 'pre-capture')

                    #---save deteksi muatan
                    if((int(cls) == 1) or (int(cls) == 2)) and (int(cls)==0) and (arah == 1):
                        str_date_time = self.get_date_time()


                        filenameSave = "ALL_Img_" + str_date_time + "_0S.jpg"
                        if OSWindows:
                            filename = os.path.join(os.getcwd() + "\\images\\", filenameSave)
                        else:
                            filename = os.path.join(os.getcwd() + "/images/", filenameSave)

                        print(filename)
                        img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                        cv2.imwrite(filename, img_resize)

                        #self.UploadIMG(filenameSave,filename, '1','pre-capture')
                        #self.UploadIMGtoPedati(filenameSave, filename, cls, 'pre-capture')


                    if(int(cls) == 0) or (int(cls) == 1) or (int(cls) == 2):
                        if (int(cls) == 0):
                            cv2.rectangle(current_frame_small, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,255,255), 2)
                            cv2.putText(current_frame_small, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                        if (int(cls) == 1):
                            cv2.rectangle(current_frame_small, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,255), 2)
                            cv2.putText(current_frame_small, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                        if (int(cls) == 2):
                            cv2.rectangle(current_frame_small, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,255,0), 2)
                            cv2.putText(current_frame_small, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

                if (cy > YlineDetect1) and (int(cls) == 1) and (skip_double == 0) and (chtruk == 1):
                    skip_double = 1
                    sabes_count += 1
                    cntFileSaveSabes +=1

                    str_date_time = self.get_date_time()

                    filenameSave = "Img_" + str_date_time + "_0S.jpg"
                    if OSWindows:
                        filename = os.path.join(os.getcwd() + "\\images\\", +filenameSave)
                    else:
                        filename = os.path.join(os.getcwd() + "/images/", +filenameSave)

                    print(filename)


                    self.filegambarstart = "preCap_" + filenameSave
                    if OSWindows:
                        self.filenamestart = os.path.join(os.getcwd() + "\\images\\", self.filegambarstart)
                    else:
                        self.filenamestart = os.path.join(os.getcwd() + "/images/", self.filegambarstart)

                    print(self.filenamestart)


                    img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                    cv2.imwrite(filename, img_resize)
                    cv2.imwrite("tempImg.jpg", current_frame_small)
                    cv2.imwrite(self.filenamestart, self.img_resizePrecap)

                    #cv2.imwrite(filename, img_resize)

                    #self.UploadIMG( filenameSave,filename, '0','recorded')
                    #self.UploadIMGtoPedati(filenameSave, filename, cls, 'recorded')
                    isExistPreImg = os.path.exists(self.filenamestart)
                    isExistCapImg = os.path.exists(filename)
                    if (isExistPreImg and isExistCapImg):
                        self.UploadIMGtoPedati(filenameSave, filename,self.filegambarstart,self.filenamestart, cls, 'recorded')

                if (cy > YlineDetect1) and (int(cls) == 2) and (skip_double == 0) and (chtruk == 1):
                    skip_double = 1
                    batu_count += 1

                    cntFileSaveBatu +=1
                    #filename = f'savedImageBatu_{cntFileSaveBatu}.jpg'
                    current_time = datetime.datetime.now()
                    tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
                    jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)
                    filenameSave = "Img_" + str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) +"_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second) + "_1B.jpg"
                    if OSWindows:
                        filename = os.path.join(os.getcwd() + "\\images\\", filenameSave)
                    else:
                        filename = os.path.join(os.getcwd() + "/images/", filenameSave)
                    print(filename)
                    print(filenameSave)

                    self.filegambarstart = "preCap_" + filenameSave
                    if OSWindows:
                        self.filenamestart = os.path.join(os.getcwd() + "\\images\\", self.filegambarstart)
                    else:
                        self.filenamestart = os.path.join(os.getcwd() + "/images/", self.filegambarstart)

                    print(self.filenamestart)

                    img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                    cv2.imwrite(filename, img_resize)
                    cv2.imwrite("tempImg.jpg", current_frame_small)
                    cv2.imwrite(self.filenamestart, self.img_resizePrecap)
                    #cv2.imwrite(filename, img_resize)

                    #self.UploadIMG(filenameSave,filename, '1','recorded')
                    #self.UploadIMGtoPedati(filenameSave, filename, cls, 'recorded')
                    isExistPreImg = os.path.exists(self.filenamestart)
                    isExistCapImg = os.path.exists(filename)
                    if (isExistPreImg and isExistCapImg):
                        self.UploadIMGtoPedati(filenameSave, filename,self.filegambarstart,self.filenamestart, cls, 'recorded')
                if (cy <= (YlineDetect1-50)) and ((int(cls) == 1) or (int(cls) == 2)):
                    skip_double = 0
                    chtruk  = 0
                    tempCy = 0
        
        #-----
        cv2image2 = cv2.cvtColor(current_frame_small, cv2.COLOR_BGR2RGBA)
        img2 = Image.fromarray(cv2image2)
        imgtk2 = ImageTk.PhotoImage(image=img2)

        self.lmain1.imgtk = imgtk2
        self.lmain1.configure(image=imgtk2)

        
    def upload_vid_func(self):
        print(imageVal)
        
        def browse_file():
            def run_yolov5_on_video():

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
                    global cntdot

                    _, frame = cap.read()
                    if frame is None:
                        return
                    else:
                        w, h = frame.shape[1],frame.shape[0]
                    current_frame_small = cv2.resize(frame,(0,0),fx=1,fy=1)
                    #current_frame_small = cv2.resize(frame,(0,0),fx=0.35,fy=0.35)
                    
                    if cntdot > self.fps:
                        cntdot = 0
                    if (cntdot % 2) == 0  :
                        results = model(current_frame_small)
                        
                        self.lbl_val['text'] = ": None"
                        
                        #self.reUPLOAD_img()
                        #self.delete_old_img()
                        cv2.line(current_frame_small, (0, YlineDetect0),(w, YlineDetect0), (0,255, 255), thickness=3)

                        #======
                        self.detect_muatan(results, current_frame_small,w)
                        
                    cntdot +=1
                    #print(cntdot)
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