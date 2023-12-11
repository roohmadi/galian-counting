# importing required libraries
import cv2 
import time

import imutils
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

global skip_double,sabes_count, batu_count, cntFileSaveSabes, cntFileSaveBatu, PINTU, img_del_date, host, weight_file, OStype
global cntdot, fps, tempCy, chtruk
global  YlineDetect0,YlineDetect1
YlineDetect1 = 230
YlineDetect0 = 100

tempCy = 0
chtruk = 0
fps = 0
cntdot = 0
skip_double = 0
sabes_count =  0
batu_count = 0
cntFileSaveSabes = 0
cntFileSaveBatu = 0
cntFlag = 0

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

model = torch.hub.load('ultralytics/yolov5', 'custom', weight_file)

# defining a helper class for implementing multi-threaded processing 
class CCTVStream :
    global fps, skip_double,sabes_count, batu_count, cntFileSaveBatu, cntFileSaveSabes, cntFlag,cntdot
    global chtruk, tempCy, arah

    def __init__(self, stream_id=0):
        self.stream_id = stream_id   # default is 0 for primary camera 
        
        # opening video capture stream 
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing cctv stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        fps = fps_input_stream
        print("FPS of cctv hardware/input stream: {}".format(fps_input_stream))
            
        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
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
                self.stopped = True
                break 
        self.vcap.release()

    # method for returning latest read frame 
    def read(self):
        return self.frame

    # method called to stop reading frames 
    def stop(self):
        self.stopped = True
    
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
                if cctv_stream.UploadIMGtoPedati(fileN,os.path.join(os.getcwd() + "\\images\\", fileN),fileN[-6],'re-upload'):
                    print("Upload offline suksess...")
            else:
                if cctv_stream.UploadIMGtoPedati(fileN,os.path.join(os.getcwd() + "/images/", fileN),fileN[-6],'re-upload'):
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

    def UploadIMGtoPedati(self,filenameSave, filename, muatan, source):
        from datetime import datetime
        if cctv_stream.connect():
            hostpedati = 'http://pedati.id:54100/mblb/api/main/insertcapture'
            now = datetime.now()
            print("now =", now)
            date_format = now.strftime("%Y-%m-%d %H:%M:%S")

            #date_obj = datetime.strptime(date_str, date_format)
            print(date_format)
            now = datetime.now()
            tgl_jam = now.strftime("%Y-%m-%d %H:%M:%S")
            dfile = open(filename, "rb").read()
            files = {'filegambar': (filenameSave, dfile, 'image/jpg', {'Expires': '0'})}

            #data = {'id_muatan': muatan, 'pintu': str(PINTU), 'tanggal_capture': tgl_jam, 'filename': filenameSave, 'filegambar': (filenameSave, dfile, 'image/jpg', {'Expires': '0'})}
            #data = {'id_muatan': muatan, 'pintu': PINTU, 'tanggal_capture': tgl_jam, 'filename': filenameSave}
            data = {'id_muatan': filenameSave[-6], 'pintu': PINTU, 'tanggal_capture': tgl_jam, 'filename': filenameSave, 'filegambar': filenameSave}
    
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
                        newfileloc = path_imgupl + "upl_" + filenameSave

                        os.rename(filename, newfilename)
                        shutil.move(newfilename,newfileloc)
                        print("sudah rename: ")

                    return True

    def UploadIMG(self,filenameSave, filename, muatan, source):
        if cctv_stream.connect():
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

                        os.rename(filename, newfilename)
                        shutil.move(newfilename,newfileloc)
                        print("sudah rename: ")

                    return True

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
                        str_date_time = self.get_date_time()
                        filenameSave = "preCap_Img_" + str_date_time + "_0S.jpg"
                        if OSWindows:
                            filename = os.path.join(os.getcwd() + "\\images\\", "res_"+filenameSave)
                        else:
                            filename = os.path.join(os.getcwd() + "/images/", "res_"+filenameSave)

                        print(filename)
                        img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                        cv2.imwrite(filename, img_resize)

                    #---save deteksi muatan
                    if((int(cls) == 1) or (int(cls) == 2)) and (int(cls)==0) and (arah == 1):
                        str_date_time = self.get_date_time()


                        filenameSave = "ALL_Img_" + str_date_time + "_0S.jpg"
                        if OSWindows:
                            filename = os.path.join(os.getcwd() + "\\images\\", "res_"+filenameSave)
                        else:
                            filename = os.path.join(os.getcwd() + "/images/", "res_"+filenameSave)

                        print(filename)
                        img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                        cv2.imwrite(filename, img_resize)

                        self.UploadIMG(filenameSave,filename, '1','pre-capture')
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
                        filename = os.path.join(os.getcwd() + "\\images\\", "res_"+filenameSave)
                    else:
                        filename = os.path.join(os.getcwd() + "/images/", "res_"+filenameSave)

                    print(filename)


                    img_resize = cv2.resize(current_frame_small,(0,0),fx=0.5,fy=0.5)
                    cv2.imwrite(filename, img_resize)
                    cv2.imwrite("tempImg.jpg", current_frame_small)

                    #cv2.imwrite(filename, img_resize)

                    self.UploadIMG( filenameSave,filename, '0','recorded')
                    #self.UploadIMGtoPedati(filenameSave, filename, cls, 'recorded')

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
                    #self.UploadIMGtoPedati(filenameSave, filename, cls, 'recorded')
                if (cy <= (YlineDetect1-50)) and ((int(cls) == 1) or (int(cls) == 2)):
                    skip_double = 0
                    chtruk  = 0
                    tempCy = 0



# initializing and starting multi-threaded webcam capture input stream
cctv_stream = CCTVStream(stream_id=val_rtsp) #  stream_id = 0 is for primary camera 
cctv_stream.start()

# processing frames in input stream
num_frames_processed = 0 
start = time.time()
while True :
    
    if cctv_stream.stopped is True :
        break
    else :
        frame = cctv_stream.read() 

    # adding a delay for simulating time taken for processing a frame 
    delay = 0.03 # delay value in seconds. so, delay=1 is equivalent to 1 second 
    time.sleep(delay) 
    num_frames_processed += 1
    
    frame = imutils.resize(frame, width=800)
    
    #-----main code --
    w, h = frame.shape[1],frame.shape[0]
    #current_frame_small = cv2.resize(frame,(0,0),fx=1,fy=1)
    
    
    if cntdot > fps:
        cntdot = 0
    if (cntdot % 10) == 0  :
        results = model(frame)

    current_frame_small = cv2.resize(frame,(0,0),fx=1,fy=1)

    #cctv_stream.reUPLOAD_img()
    #cctv_stream.delete_old_img()
    cv2.line(current_frame_small, (0, YlineDetect0),(w, YlineDetect0), (0,255, 255), thickness=3)

    #======
    cctv_stream.detect_muatan(results, current_frame_small,w)


    #-----end main code ---

    cv2.imshow('frame' , current_frame_small)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
end = time.time()
cctv_stream.stop() # stop the webcam stream 

# printing time elapsed and fps 
#elapsed = end-start
#fps = num_frames_processed/elapsed 
#print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))

# closing all windows 
cv2.destroyAllWindows()
