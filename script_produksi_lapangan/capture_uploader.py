# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 21:52:06 2024

@author: roohm
"""

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

#---tambahan
#from yolov5.models.experimental import attempt_load
#from yolov5.utils.downloads import attempt_download
#from yolov5.models.common import DetectMultiBackend
#from yolov5.utils.dataloaders import LoadImages, LoadStreams
#from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes,
#                                  check_imshow, xyxy2xywh, increment_path)
#from yolov5.utils.torch_utils import select_device, time_sync
#from yolov5.utils.plots import Annotator, colors
#from deep_sort.utils.parser import get_config
#from deep_sort.deep_sort import DeepSort

from tkinter import filedialog as fd
import datetime
from datetime import date

global  YlineDetect0,YlineDetect1, Y0, Y1, cyTruk
YlineDetect1 = 420
YlineDetect0 = 170
Y0 = 40
Y1 = 200
cyTruk = 0

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
global cntOBJ, arah
global captureOK, diff
global jam
jam = 0
diff = 0
captureOK = 0
cntOBJ = 0
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
arah = 0




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
    path_detect = os.getcwd()+"\\runs\\detect\\"
else:
    #--Linux
    path_img = os.getcwd()+"/images/"
    path_imgTemp = os.getcwd()+"/tempImg.jpg"
    path_imgTempDetect = os.getcwd()+'/'+"tempImgDetect.jpg"
    path_imgupl = os.getcwd()+"/imgupl/"
    path_log = os.getcwd()+"/log/"
    path_detect = os.getcwd()+"/runs/detect/"

fileLog = path_log + str_tgl + '.logU'
print('--------=---------')
print("image saved to " + path_img)
print("cctv: " + val_rtsp)

isExistlog = os.path.exists(path_log)
if isExistlog:
    print("folder log exist")
else:
    print("folder log not exist")
    os.mkdir(path_log)


#PINTU = 3

# Load YOLOv5 model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#print(model)
#model = torch.hub.load('ultralytics/yolov5', 'custom', 'galian_200epch_1k5.pt')



class App:
    def __init__(self, window, window_title, video_source=val_rtsp):


        self.window = window
        self.window.title(window_title)
        window_height = 350
        window_width = 300

        screen_width = self.window .winfo_screenwidth()
        screen_height = self.window .winfo_screenheight()

        x_cordinate = int((screen_width/2) - (window_width/2))
        y_cordinate = int((screen_height/2) - (window_height/2))


        self.window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

        msgLog = '....................'
        self.msgtoLog(fileLog,msgLog)
        msgLog = 'Starting application'
        self.msgtoLog(fileLog,msgLog)



        self.window.rowconfigure(0, minsize=100, weight=1)
        self.window.columnconfigure(1, minsize=200, weight=1)

        fr_button = Frame(self.window)
        fr_graph = Frame(self.window)
        fr_result = Frame(self.window)

        # -------
        #self.btn_rdcsv = Button(fr_button, text="Upload Image", command = self.upload_img_func)
        #self.btn_wrcsv = Button(fr_button, text="Upload Video",command = self.upload_vid_func)
        #self.btn_baseline = Button(fr_button, text="Open CCTV",)
        self.btn_quit = Button(fr_button, text="Quit", command=self.window.destroy)

        #self.btn_rdcsv.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        #self.btn_wrcsv.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        #self.btn_baseline.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        self.btn_quit.grid(row=0, column=1, sticky="ew", padx=10, pady=10)

        #self.btn_rdcsv.config(width=10, height=5)
        #self.btn_wrcsv.config(width=10, height=5)
        #self.btn_baseline.config(width=10, height=5)
        self.btn_quit.config(width=10, height=5)
        #self.btn_rdcsv["state"] = DISABLED
        #self.btn_baseline["state"] = DISABLED

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

        self.lbl_phase_max = Label(fr_button, text="Object local", font=('Times 14'))
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

        self.window.after(2000, self.timer)

        self.window.mainloop()

    def connect(self,host='http://google.com'):
        try:
            urllib.request.urlopen(host) #Python 3.x
            return True
        except:
            return False

    def date_filename (self):
        current_time = datetime.datetime.now()        
        str_tgl = str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day)
        return str(str_tgl)
    
    def get_date_time_file (self):
        current_time = datetime.datetime.now()
        #print(current_time)
        tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
        #print(tgl)
        jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)
        #print(jam)
        str_date_time = str(current_time.year) + "/" + str(current_time.month) + "/" + str(current_time.day) +" " + str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)
        
        #print(str_date_time)
        return str_date_time

    def msgtoLog (self,file, msg):
        #fileLog = path_log + file
        print(file)
        isExistLogFile = os.path.exists(file)
        if isExistLogFile:
            f = open(file, "a")
            f.write(str(self.get_date_time_file()) +  ": " + msg +"\n")
            f.close()
        else:
            f = open(file, "a")
            f.write(str(self.get_date_time_file()) +  ": " + msg +"\n")
            f.close()

    def timer(self):
        global jam

        #print(datetime.datetime.now())
        if self.connect():
            self.lbl_rtsp1val['text'] = ": Connected"
        else:
            msgLog = "Connection loss"
            self.msgtoLog(fileLog,msgLog)
            self.lbl_rtsp1val['text'] = ": Connection loss"
        cnt, cntS, cntB = self.count_local()
        self.lbl_val['text'] = ": " + str(cnt) + " not uploaded"
        self.lbl_sabes['text'] = ": " + str(cntS) + " not uploaded"
        self.lbl_batu['text'] = ": " + str(cntB) + " not uploaded"

        if cnt > 0:
            print("upload")
            self.reUPLOAD_img()
        self.delete_old_img()

        current_time = datetime.datetime.now()
        if jam != current_time.hour:
            jam = current_time.hour
            if self.connect():
                msgLog = "Connected"
            else:
                msgLog = "Connection loss"
            self.msgtoLog(fileLog,msgLog)

        self.window.after(2000, self.timer)

    def count_local (self):
        cnt = 0
        cntS = 0
        cntB = 0
        #imgOFF = os.listdir(path_img)
        #cek_img_off = len(imgOFF)
        for fileN in os.listdir(path_img):
            x = fileN.split("_")
            if (x[0] == 'Img'):
                cnt +=1
                if (fileN[-6] == '0'):
                    cntS +=1
                if (fileN[-6] == '1'):
                    cntB +=1

        return cnt, cntS, cntB

    def reUPLOAD_img(self):
        #print(path_img)

        imgOFF = os.listdir(path_img)
        cek_img_off = len(imgOFF)
        #print(cek_img_off)
        if cek_img_off > 0:
            fileN = imgOFF[0]
            x = fileN.split("_")
            if (x[0] == 'Img'):
                filenameSave = fileN
                filegambarstart = "preCap_"+filenameSave
            if (x[0] == 'preCap'):
                pan=len(fileN)
                filenameSave = fileN[7:pan]
                filegambarstart = fileN
                print(filenameSave)
                print(filegambarstart)
            if OSWindows:
                filename = path_img + "\\" + filenameSave
                filenamestart = path_img + "\\" + filegambarstart

                if self.UploadIMGtoPedatiDouble(filenameSave, filename,filegambarstart, filenamestart, filename[-6], 'live'):
                #if UploadIMGtoPedatiDouble(fileN,os.path.join(os.getcwd() + "\\images\\", fileN),fileN[-6],'re-upload'):
                    print("Upload offline suksess...")
                    if (int(filename[-6]) == 0 ):
                        msgLog = "Upload sabes offline suksess..."
                        self.msgtoLog(fileLog,msgLog)
                    if (int(filename[-6]) == 1 ):
                        msgLog = "Upload batu belah offline suksess..."
                        self.msgtoLog(fileLog,msgLog)
            else:
                #if UploadIMGtoPedatiDouble(fileN,os.path.join(os.getcwd() + "/images/", fileN),fileN[-6],'re-upload'):
                filename = path_img + filenameSave
                filenamestart = path_img  + filegambarstart
                if self.UploadIMGtoPedatiDouble(filenameSave, filename,filegambarstart, filenamestart, filename[-6], 'live'):
                    print("Upload offline suksess...")
                    if (int(filename[-6]) == 0 ):
                        msgLog = "Upload sabes offline suksess..."
                        self.msgtoLog(fileLog,msgLog)
                    if (int(filename[-6]) == 1 ):
                        msgLog = "Upload batu belah offline suksess..."
                        self.msgtoLog(fileLog,msgLog)

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
                if (x[1] == 'Img'):
                    date_create = date(int(x[2]),int(x[3]),int(x[4]))
                    get_diff_days = (date.today() - date_create).days
                if (x[1] == 'preCap') : 
                    date_create = date(int(x[3]),int(x[4]),int(x[5]))
                    get_diff_days = (date.today() - date_create).days               


                if get_diff_days > img_del_date:
                    filePre = fileN[:4] + "preCap_" + fileN[4:]
                    #print(path_img + fileN)
                    isExistdelUPL = os.path.exists(path_imgupl + fileN)
                    isExistdelPre = os.path.exists(path_imgupl + filePre)
                    #print(isExistdelUPL)
                    if isExistdelUPL:
                        os.remove(path_imgupl +"/"+ fileN)
                        if isExistdelPre:
                            os.remove(path_imgupl +"/"+ filePre)
                        isExistdelUPL = os.path.exists(path_imgupl + fileN)
                        if isExistdelUPL:
                            print("file " + fileN + " gagal dihapus")
                        else:
                            print("file " + fileN + " telah dihapus")
                            msgLog = "file " + fileN + " telah dihapus"
                            self.msgtoLog(fileLog,msgLog)
                                #else:
                                    #    print("NONE")
    def UploadIMGtoPedatiDouble(self,filenameSave, filename,filegambarstart,filenamestart, muatan, source):
        from datetime import datetime
        if self.connect():
            hostpedati = 'http://pedati.id:54100/mblb/api/main/insertcapturedouble'
            #hostpedati = 'http://pedati.id:54100/mblb/api/main/insertcapture'
            now = datetime.now()
            print("now =", now)
            date_format = now.strftime("%Y-%m-%d %H:%M:%S")

            #date_obj = datetime.strptime(date_str, date_format)
            print(date_format)
            now = datetime.now()

            #tgl_jam = now.strftime("%Y-%m-%d %H:%M:%S")
            print(filename)
            print(filenamestart)
            isExistTempImg = os.path.exists(filename)
            isExistPre = os.path.exists(filenamestart)
            print(isExistTempImg)
            print(isExistPre)
            
            
            if isExistTempImg:
                x = filename.split("_")
                my_date = x[1] + "-" + x[2]+"-"+x[3] + " " + x[4] + ":" + x[5]+":"+x[6]
                tgl_jam=datetime.strptime(my_date, "%Y-%m-%d %H:%M:%S")
                dfile = open(filename, "rb").read()
            
            if isExistPre:
                dfile1 = open(filenamestart, "rb").read()
                #files = {'filegambar': (filenameSave, dfile, 'image/jpg', {'Expires': '0'})}
                files = {'filegambar': (filenameSave, dfile, 'image/jpg', {'Expires': '0'}),'filegambarstart': (filegambarstart, dfile1, 'image/jpg', {'Expires': '0'})}

                #data = {'id_muatan': muatan, 'pintu': str(PINTU), 'tanggal_capture': tgl_jam, 'filename': filenameSave, 'filegambar': (filenameSave, dfile, 'image/jpg', {'Expires': '0'})}
                #data = {'id_muatan': muatan, 'pintu': PINTU, 'tanggal_capture': tgl_jam, 'filename': filenameSave}
                #data = {'id_muatan': filenameSave[-6], 'pintu': PINTU, 'tanggal_capture': tgl_jam, 'filename': filenameSave, 'filegambar': filenameSave}
                data = {'id_muatan': str(int(filenameSave[-6])+1), 'pintu': PINTU, 'tanggal_capture': tgl_jam, 'filename': filenameSave, 'filegambar': filenameSave,'filenamestart': filegambarstart, 'filegambarstart': filegambarstart}

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
                            newfilenamePre = path_img + "upl_" +  filegambarstart
                            newfileloc = path_imgupl + "/upl_" + filenameSave
                            newfilelocPre = path_imgupl + "/upl_" +  filegambarstart
                            print(newfilename)
                            print(newfilenamePre)
                            print(newfileloc)
                            print(newfilelocPre)

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