# importing required libraries
import cv2 
import time

#import imutils
from threading import Thread # library for implementing multi-threaded processing 

#---import main
import datetime
from datetime import date
import urllib.request
import os
import torch
import shutil
from os.path import exists
import requests
import numpy as np

#import pathlib
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

from skimage.metrics import structural_similarity as ssim

#---logger
import logging


global skip_double,sabes_count, batu_count, cntFileSaveSabes, cntFileSaveBatu, PINTU, img_del_date, host, weight_file, OStype
global cntdot, tempCy, chtruk, skip_double0, chtruk0
global cntOBJ, arah
global captureOK, diff
global  YlineDetect0,YlineDetect1, Y0, Y1, cyTruk
global time_pro,jam,temp_diff
global kelas, confNew, mse_threshold

mse_threshold = 400
temp_diff = 0
jam = 0
time_pro = 0
YlineDetect1 = 350
YlineDetect0 = 130 #170
Y0 = 20 #40
Y1 = 200
skip_double = 0
chtruk  = 0
tempCy = 0
confNew = 0
diff = 0
captureOK = 0
cntOBJ = 0
chtruk0 = 0
cyTruk = 0

cntdot = 0
tempCy = 0
skip_double0 = 0

sabes_count =  0
batu_count = 0
cntFileSaveSabes = 0
cntFileSaveBatu = 0
cntFlag = 0
arah = 0

import warnings
warnings.filterwarnings("ignore")
running = False
imageVal = True
saveTempImgFlag = False

production = False



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
print("weight: " + weight_file)
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

fileLog = path_log + str_tgl + '.log'
print('--------=---------')
print("image saved to " + path_img)
print("cctv: " + val_rtsp)

isExistlog = os.path.exists(path_log)
if isExistlog:
    print("folder upl exist")
else:
    print("folder upl not exist")
    os.mkdir(path_log)
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
    


#model = torch.hub.load('ultralytics/yolov5', 'custom', weight_file)
if OSWindows:
    #model = torch.load(os.getcwd()+ '\\' +weight_file)
    model = torch.hub.load(os.getcwd()+ '\\yolov5', 'custom', path=os.getcwd()+ '\\' +weight_file, source='local')  # local repo
else:
    model = torch.hub.load(os.getcwd()+ '/yolov5', 'custom', path=os.getcwd()+ '/' +weight_file, source='local')  # local repo
    

# defining a helper class for implementing multi-threaded processing 
class CCTVStream :
    global fps, skip_double,sabes_count, batu_count, cntFileSaveBatu, cntFileSaveSabes, cntFlag,cntdot
    global chtruk, tempCy, arah

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

    def __init__(self, stream_id=0):
        #logging.config.fileConfig(fname='logger.ini')
        str_tgl = CCTVStream.date_filename (self)
        #fileLog = str_tgl + '.log'
        msgLog = '....................'
        CCTVStream.msgtoLog(self,fileLog,msgLog)
        msgLog = 'Starting application'
        CCTVStream.msgtoLog(self,fileLog,msgLog)

        self.stream_id = stream_id   # default is 0 for primary camera
        #self.stream_id = 'C:\\Users\\roohm\\vehicle-counting-yolov5\\data\\SABES_26012024.mp4'
        #print(self.stream_id)

        # opening video capture stream 
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing cctv stream.")
            msgLog = '[Exiting]: Error accessing cctv stream.'
            CCTVStream.msgtoLog(self,fileLog,msgLog)
            #cctv_stream.msgtoLog(fileLog,msgLog)
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        fps = fps_input_stream
        print("FPS of cctv hardware/input stream: {}".format(fps_input_stream))
        msgLog = 'Starting, CCTV live'
        CCTVStream.msgtoLog(self,fileLog,msgLog)
        #cctv_stream.msgtoLog(fileLog,msgLog)
            
        # reading a single frame from vcap stream for initializing

        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            msgLog = '[Exiting] No more frames to read'
            CCTVStream.msgtoLog(self,fileLog,msgLog)
            #cctv_stream.msgtoLog(fileLog,msgLog)
            #exit(0)
            time.sleep(0.01)

        # self.stopped is set to False when frames are being read from self.vcap stream 
        self.stopped = True 

        # reference to the thread for reading next available frame from input stream 
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads keep running in the background while the program is executing 
        
        
        
        
        
    # method for starting the thread for grabbing next available frame in input stream 
    def start(self):
        self.stopped = False
        self.t.start() 

    # method for reading next frame 
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('UPDATE: [Exiting] No more frames to read')
                msgLog = 'UPDATE: [Exiting] No more frames to read'
                #CCTVStream.msgtoLog(self,fileLog,msgLog)
                cctv_stream.msgtoLog(fileLog,msgLog)
                cv2.putText(current_frame_small,  "Out " + msgLog,((50),h-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                self.stopped = True
                break 
        self.vcap.release()

    # method for returning latest read frame 
    def read(self):
        return self.frame

    # method called to stop reading frames 
    def stop(self):
        self.stopped = True

    def mse(self, imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err


    def detect_muatan(self, results, current_frame_small,img_ori,w,h):
        global skip_double,sabes_count, batu_count, cntFileSaveBatu, cntFileSaveSabes, cntFlag
        global chtruk, tempCy, arah,cyTruk
        global skip_double0, chtruk0
        global captureOK
        global diff,temp_diff
        global kelas, confNew, mse_threshold, m, s
        #print("detect muatan")
        cntOBJ = 0
        m = 0
        s = 0
        cv2.putText(img_ori,  "PINTU " + str(PINTU),(((w//2)-50),40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        for *xyxy, conf, cls in results.xyxy[0]:
            cx, cy = (int(xyxy[0]+(xyxy[2]-xyxy[0])/2),
                  int(xyxy[1]+(xyxy[3]-xyxy[1])/2))
            #print("cy: " + str(cy) + "   cyTemp: " + str(tempCy))

            if (int(cls)==0):
                cyTruk = cy
            label = f'{model.names[int(cls)]} {conf:.2f} {cls}'
            if (int(cls)>0):
                if ((conf) > (confNew)) and (int(cls) < 3):
                    confNew = (conf)
                    print(confNew)
                print(label)
                print("cyTruk: " + str(cyTruk))
            msgLog = label
            end = time.time()
            elapsed = end-start
            time_pro = elapsed
            


            cv2.putText(img_ori,  "Process time: " + str(round(elapsed,2)) + " s/frame",(450,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
            if (int(cls)==1):
                cv2.rectangle(img_ori, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,255), 2)
                cv2.putText(img_ori, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            elif (int(cls)==2):
                cv2.rectangle(img_ori, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,255,0), 2)
                cv2.putText(img_ori, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

            if (int(cls)==0):
                diff = tempCy - cyTruk
                if ((diff-temp_diff) < 0):
                    arah = 1
                if ((diff-temp_diff) > 0):
                    arah = 0
                temp_diff = diff

            #print("cyTruk:" + str(cyTruk) + " chtruk: " + str(chtruk) + " skip_double: " + str(skip_double) + " arah: " +str(arah))
            #print(str((cyTruk > (YlineDetect0))) + " - " + str((cyTruk < YlineDetect1)) + " - " + str((skip_double == 0)) + "=" + str((cyTruk > (YlineDetect0)) and (cyTruk < YlineDetect1) and (skip_double == 0)))
            #print(str(cyTruk > (YlineDetect0)) + " - " + str((skip_double == 0)) + " - " + str((chtruk == 1)) + " - " + str((arah == 1)) + "=" + str((cyTruk > (YlineDetect0)) and (skip_double == 0) and (chtruk == 1) and (arah == 1)))
            #print()
            #print()
            if (arah == 1):
                cv2.circle(current_frame_small, (30,h//2), 10, (0,255,0), 20)
            if (arah == 0):
                cv2.circle(current_frame_small, (30,h//2), 10, (0,0,255), 20)
            if (confNew > 0.65) and (int(cls) > 0) and (int(cls) < 5):
                if (cyTruk > (YlineDetect0)) and (cyTruk < YlineDetect1) and (skip_double == 0):
                    chtruk  = 1
                if cyTruk > (YlineDetect0) and (skip_double == 0) and (chtruk == 1) and (arah == 1) :
                    if (int(cls) == 1):
                        kelas = '1S'
                    if (int(cls) == 2):
                        kelas = '2B'
                    if (int(cls) == 4):
                        kelas = '4K'
                    #else:
                        #kelas = '0X'
                    current_time = datetime.datetime.now()
                    tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
                    jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)
                    filenameSave = "Img_" + str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) +"_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second) + "_"+ kelas +".jpg"

                    if OSWindows:
                        filename = os.path.join(os.getcwd() + "\\images\\", filenameSave)
                    else:
                        filename = os.path.join(os.getcwd() + "/images/", filenameSave)

                    print(filename)
                    if ((int(cls) == 1) or (int(cls) == 2)):

                        isExistTemp = os.path.isfile('last_Img.jpg')
                        print('')
                        print("isExistTemp")
                        print(isExistTemp)
                        print('')
                        if (isExistTemp):
                            imageA = cv2.imread("last_Img.jpg")
                            imageB = current_frame_small
                            # convert the images to grayscale
                            imageA_GRY = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
                            imageB_GRY = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

                            #("Img1 Resolution:", imageA_GRY.shape)
                            #print("Img2 Resolution:", imageB_GRY.shape)

                            m = cctv_stream.mse(imageA_GRY, imageB_GRY)
                            s = ssim(imageA_GRY, imageB_GRY)

                            label_m = f'{m:.2f}'
                            label_s = f'{s:.2f}'
                            #print(label_m)
                            cv2.putText(img_ori,"MSE: " + str(label_m), (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                            cv2.putText(img_ori,"SSIM: " + str(label_s), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                        else:
                            cv2.imwrite('last_Img.jpg', current_frame_small)

                        isExistLastImg = os.path.isfile('last_Img.jpg')
                        print("isExistLastImg")
                        print(isExistLastImg)
                        if (isExistLastImg):
                            os.remove("last_Img.jpg")
                            cv2.imwrite('last_Img.jpg', current_frame_small)

                        mse_threshold = 400
                        if m > mse_threshold:
                            cv2.imwrite(filename, current_frame_small)
                            msgLog = 'Object detected, save the image, process time: ' + str(time_pro) + ' s/frame'


                        #cv2.imwrite(filename, current_frame_small)
                        #msgLog = 'Object detected, save the image, process time: ' + str(time_pro) + ' s/frame'
                            cctv_stream.msgtoLog(fileLog,msgLog)
                            print(label)
                            print("")
                            print("")
                            print("")
                        skip_double = 1
                print("detected: " + str(int(cls)) + " chtruk: " + str(chtruk) + " skip_double: " + str(skip_double) + " arah: " +str(arah))
            if (cyTruk >= (YlineDetect1))  and ((int(cls) < 5)) and (skip_double == 1):
                skip_double = 0
                chtruk  = 0
                tempCy = 0
                confNew = 0
                print("RESET: " + str(int(cls)) + " chtruk: " + str(chtruk) + " skip_double: " + str(skip_double) + " arah: " +str(arah))
            if (cyTruk <= (YlineDetect0))  and ((int(cls) < 5)) and (skip_double == 1):
                skip_double = 0
                chtruk  = 0
                tempCy = 0
                confNew = 0
                print("RESET: " + str(int(cls)) + " chtruk: " + str(chtruk) + " skip_double: " + str(skip_double) + " arah: " +str(arah))





# initializing and starting multi-threaded webcam capture input stream
cctv_stream = CCTVStream(stream_id=val_rtsp) #  stream_id = 0 is for primary camera
#cctv_stream = CCTVStream(stream_id="C:\Users\roohm\vehicle-counting-yolov5\data\SABES_26012024.mp4")
cctv_stream.start()
#cap = cv2.VideoCapture(file_path)

# processing frames in input stream
num_frames_processed = 0 

while True :
    
    if cctv_stream.stopped is True :
        break
    else :
        frame = cctv_stream.read() 

    # adding a delay for simulating time taken for processing a frame 
    delay = 0.03 # delay value in seconds. so, delay=1 is equivalent to 1 second 
    time.sleep(delay) 
    num_frames_processed += 1
    
    #frame = imutils.resize(frame, width=800)
    #frame = cv2.resize(frame,(0,0),fx=1,fy=1)
    
    #-----main code --

    current_frame_small = cv2.resize(frame,(0,0),fx=1.3,fy=1.3)
    w, h = current_frame_small.shape[1],current_frame_small.shape[0]

    
    if cntdot > 20:
        cntdot = 0
    if (cntdot % 10) == 0  :
        start = time.time()
        crop_img = current_frame_small[YlineDetect0:YlineDetect1, 0:w]
        #results = model(crop_img)
        results = model(current_frame_small)


    #======
    #           detect_muatan(results, crop_img,current_frame_small,w,h)
    cctv_stream.detect_muatan(results, current_frame_small, current_frame_small,w,h)
    cv2.line(current_frame_small, (0, YlineDetect0),(w, YlineDetect0), (0,255, 255), thickness=3)
    cv2.line(current_frame_small, (0, YlineDetect1),(w, YlineDetect1), (0,255, 255), thickness=3)
    #cv2.line(crop_img, (0, Y0),(w, Y0), (255,255, 0), thickness=3)
    #cv2.line(crop_img, (0, Y1),(w, Y1), (255,255, 0), thickness=3)
    #cv2.line(current_frame_small, (0, 180),(w, 180), (255,0, 255), thickness=3)
    #cv2.line(current_frame_small, (0, 250),(w, 250), (255,0, 255), thickness=3)

    #-----end main code ---
    end = time.time()
    elapsed = end-start
    time_pro = elapsed
    #print("Elapsed Time: " + str(elapsed))
    cv2.putText(current_frame_small,  "Process time: " + str(round(elapsed,2)) + " s/frame",(450,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
    cv2.imshow('frame>>' , current_frame_small)
    current_time = datetime.datetime.now()
    if jam != current_time.hour:
        jam = current_time.hour
        msgLog = "Process time: " + str(round(elapsed,2)) + " s/frame"
        cctv_stream.msgtoLog(fileLog,msgLog)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cctv_stream.stop() # stop the webcam stream 

# printing time elapsed and fps 
#elapsed = end-start
#fps = num_frames_processed/elapsed 
#print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))
msgLog = 'Stop the application'
cctv_stream.msgtoLog(fileLog,msgLog)
#cv2.putText(current_frame_small, "Out: " + msgLog (50,h-50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
cv2.putText(current_frame_small,  "Out " + msgLog,((50),h-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
# closing all windows 
cv2.destroyAllWindows()
