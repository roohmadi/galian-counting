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

#---logger
import logging


global skip_double,sabes_count, batu_count, cntFileSaveSabes, cntFileSaveBatu, PINTU, img_del_date, host, weight_file, OStype
global cntdot, tempCy, chtruk, skip_double0, chtruk0
global cntOBJ, arah
global captureOK, diff
global  YlineDetect0,YlineDetect1, Y0, Y1, cyTruk
global time_pro,jam
jam = 0
time_pro = 0
YlineDetect1 = 450
YlineDetect0 = 130 #170
Y0 = 20 #40
Y1 = 200
cyTruk = 0
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
    


model = torch.hub.load('ultralytics/yolov5', 'custom', weight_file)


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
            exit(0)

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
                print('[Exiting] No more frames to read')
                msgLog = '[Exiting] No more frames to read'
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
    
    


    def detect_muatan(self, results, current_frame_small,img_ori,w,h):
        global skip_double,sabes_count, batu_count, cntFileSaveBatu, cntFileSaveSabes, cntFlag
        global chtruk, tempCy, arah,cyTruk
        global skip_double0, chtruk0
        global captureOK
        global diff
        #print("detect muatan")
        cntOBJ = 0
        for *xyxy, conf, cls in results.xyxy[0]:
            cx, cy = (int(xyxy[0]+(xyxy[2]-xyxy[0])/2),
                  int(xyxy[1]+(xyxy[3]-xyxy[1])/2))
            #print("cy: " + str(cy) + "   cyTemp: " + str(tempCy))

            label = f'{model.names[int(cls)]} {conf:.2f} {cls}'
            print(label)
            msgLog = label
            #cv2.putText(img_ori,  "Log>> " + msgLog,((20),h-20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)
            end = time.time()
            elapsed = end-start
            time_pro = elapsed
            print("Elapsed Time: " + str(elapsed))
            cv2.putText(img_ori,  "Process time: " + str(round(elapsed,2)) + " s/frame",(450,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),2)

            if conf>0.65:
                if (int(cls)==0):
                    diff = tempCy - cy
                if diff<0:
                    arah=1
                else:
                    arah=0
                if (int(cls)==1):
                    cv2.rectangle(current_frame_small, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,255), 2)
                    cv2.putText(current_frame_small, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                elif (int(cls)==2):
                    cv2.rectangle(current_frame_small, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,255,0), 2)
                    cv2.putText(current_frame_small, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                if (cy > (Y0)) and (cy < Y1) and (int(cls) == 0) and (skip_double0 == 0):
                    chtruk  = 1

                #if (cy >= 0) and ((cls == 1) or (cls == 2)) and (skip_double0 == 0):
                #    print("Capture awal")
                #    cv2.imwrite("temp.jpg", img_ori)

                if cy > (Y0) and (skip_double == 0) and (chtruk == 1) and (arah == 1) :
                    if cls==1:
                        skip_double = 1
                        sabes_count += 1

                        current_time = datetime.datetime.now()
                        tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
                        jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)
                        filenameSave = "Img_" + str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) +"_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second) + "_1B.jpg"

                        if OSWindows:
                            filename = os.path.join(os.getcwd() + "\\images\\", filenameSave)
                        else:
                            filename = os.path.join(os.getcwd() + "/images/", filenameSave)

                        print(filename)
                        filegambarstart = "preCap_" + filenameSave
                        if OSWindows:
                            filenamestart = os.path.join(os.getcwd() + "\\images\\", filegambarstart)
                        else:
                            filenamestart = os.path.join(os.getcwd() + "/images/", filegambarstart)

                        print(filenamestart)
                        #img_resize = cv2.resize(img_ori,(0,0),fx=0.5,fy=0.5)
                        cv2.imwrite(filename, img_ori)
                        cv2.imwrite(filenamestart, img_ori)
                        print("Sabes")
                        print("")
                        print("")
                        print("")
                        cv2.rectangle(current_frame_small, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,255), 2)
                        cv2.putText(current_frame_small, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                        msgLog = 'Sabes detected, save the image, process time: ' + str(time_pro) + ' s/frame'
                        #CCTVStream.msgtoLog(self,fileLog,msgLog)
                        cctv_stream.msgtoLog(fileLog,msgLog)
                    elif cls==2:
                        skip_double = 1
                        batu_count += 1
                        current_time = datetime.datetime.now()
                        tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
                        jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)
                        filenameSave = "Img_" + str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) +"_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second) + "_0S.jpg"

                        if OSWindows:
                            filename = os.path.join(os.getcwd() + "\\images\\", filenameSave)
                        else:
                            filename = os.path.join(os.getcwd() + "/images/", filenameSave)

                        print(filename)
                        filegambarstart = "preCap_" + filenameSave
                        if OSWindows:
                            filenamestart = os.path.join(os.getcwd() + "\\images\\", filegambarstart)
                        else:
                            filenamestart = os.path.join(os.getcwd() + "/images/", filegambarstart)

                        print(filenamestart)
                        #img_resize = cv2.resize(img_ori,(0,0),fx=0.5,fy=0.5)
                        cv2.imwrite(filename, img_ori)
                        cv2.imwrite(filenamestart, img_ori)
                        print("Batu Belah")
                        print("")
                        print("")
                        print("")
                        cv2.rectangle(current_frame_small, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255,255,0), 2)
                        cv2.putText(current_frame_small, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                        msgLog = 'Batu belah detected, save the image, process time: ' + str(time_pro) + ' s/frame'
                        #CCTVStream.msgtoLog(self,fileLog,msgLog)
                        cctv_stream.msgtoLog(fileLog,msgLog)
                if (cy >= (Y1)) and ((int(cls) == 0) or(int(cls) == 1) or (int(cls) == 2)):
                    skip_double = 0
                    chtruk  = 0
                    tempCy = 0

                #cv2.putText(img_ori, "cls " + str(cls) + " - " + str(cy),(50,cy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)


        #print("cy: " + str(cyTruk) + " Y1: "+ str(Y1)+"   cyTemp: " + str(tempCy) + " arah: " +str(arah) + " chtruk: " +str(chtruk) + " cap: " +str(captureOK) + " skp: " + str(skip_double))

        cv2.putText(img_ori,  "PINTU " + str(PINTU),(((w//2)-50),40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        #cv2image2 = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGBA)




# initializing and starting multi-threaded webcam capture input stream
cctv_stream = CCTVStream(stream_id=val_rtsp) #  stream_id = 0 is for primary camera
#cctv_stream = CCTVStream(stream_id=0)
cctv_stream.start()

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
        results = model(crop_img)

    #current_frame_small = cv2.resize(frame,(0,0),fx=1,fy=1)

    #cctv_stream.reUPLOAD_img()
    #cctv_stream.delete_old_img()
    #cv2.line(current_frame_small, (0, YlineDetect0),(w, YlineDetect0), (0,255, 255), thickness=3)
    cv2.line(crop_img, (0, Y0),(w, Y0), (0,255, 255), thickness=3)
    cv2.line(crop_img, (0, Y1),(w, Y1), (0,255, 255), thickness=3)

    #======
    #           detect_muatan(results, crop_img,current_frame_small,w,h)
    cctv_stream.detect_muatan(results, crop_img, current_frame_small,w,h)


    #-----end main code ---
    end = time.time()
    elapsed = end-start
    time_pro = elapsed
    print("Elapsed Time: " + str(elapsed))
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
