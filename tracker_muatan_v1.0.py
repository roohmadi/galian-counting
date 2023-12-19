"""
Author: Mahimai Raja J
Social : https://www.linkedin.com/in/mahimairaja/
This script is used to count the number of vehicles from a camera feed.
"""

import time
import datetime
from datetime import date
import urllib.request
import requests
#------

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

global cntFr, lineDetect
cntFr = 0
#----h-180
lineDetect = 360-200

up_count = 0
down_count = 0
car_count = 0
truck_count = 0
tracker1 = []
tracker2 = []

dir_data = {}

#-----
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
#-----

def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.weights, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    global cntFr

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        # print(f"Image: {img.shape} ")
        # print(f"Image Type: {type(img)} ")
        
        t1 = time_sync()


        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        if cntFr >20:
            cntFr=0
        cntFr +=1
        #print("cntFr: " + str(cntFr))
        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        if (cntFr % 1) == 0:
            pred = model(img, augment=opt.augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
            dt[2] += time_sync() - t3

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                start_time = time.time()
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    im0 = cv2.resize(im,(0,0),fx=0.25,fy=0.25)

                p = Path(p)
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                s += '%gx%g ' % img.shape[2:]

                annotator = Annotator(im0, line_width=2, pil=not ascii)
                w, h = im0.shape[1],im0.shape[0]
                # print(f"W: {w} h {h}")F
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]
                    #print("det[:, 0:4]")
                    #print(det[:, 0:4])

                    # pass detections to deepsort
                    t4 = time_sync()
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_sync()
                    dt[3] += t5 - t4

                    # draw boxes for visualization
                    #print(len(outputs))
                
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):

                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            # print(f"Img: {im0.shape}\n")
                            _dir =  direction(id,bboxes[1])
                            c = int(cls)  # integer class

                            #count
                            count_obj(bboxes,w,h,id,_dir,int(cls),im0)
                            # print(im0.shape)

                            label = f'{id} {names[c]} {conf:.2f}'
                            #print(label)
                            annotator.box_label(bboxes, label, color=colors(c, True))

                            if save_txt:
                                cv2.imwrite(txt_file_name+".jpg", im0)
                                # to MOT format
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                # Write MOT compliant results to file
                                with open(txt_path, 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                    #LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                
                    

                else:
                    deepsort.increment_ages()
                    #LOGGER.info('No detections')

                # Stream results
                im0 = annotator.result()
                if show_vid:
                    global up_count,down_count
                    color=(0,0,255)
                    # print(f"Shape: {im0.shape}")
                    #print(h)
                    #print(h-180)
                    #print(w)
                    print("Sabes: " + str(car_count) + " Batu: "+str(truck_count))

                    # Left Lane Line
                    cv2.line(im0, (0, lineDetect), (w, lineDetect), (255,0,0), thickness=2)

                    # Right Lane Line
                    #cv2.line(im0,(680,h-300),(w,h-300),(0,0,255),thickness=3)
                
                    thickness = 2 # font thickness
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.5
                    #cv2.putText(im0, "Outgoing Traffic:  "+str(up_count), (60, 150), font,
                    #            fontScale, (0,0,255), thickness, cv2.LINE_AA)

                    #cv2.putText(im0, "Incoming Traffic:  "+str(down_count), (700,150), font,
                    #    fontScale, (255,0,0), thickness, cv2.LINE_AA)
                
                # -- Uncomment the below lines to computer car and truck count --
                # It is the count of both incoming and outgoing vehicles 
                
                #Objects 
                # cv2.putText(im0, "Cars:  "+str(car_count), (60, 250), font, 
                #    1.5, (20,255,0), 3, cv2.LINE_AA)                

                # cv2.putText(im0, "Trcuks:  "+str(truck_count), (60, 350), font, 
                #    1.5, (20,255,0), 3, cv2.LINE_AA)  
                
                    end_time = time.time()
                    #fps = 1 / (end_time - start_time)
                    #cv2.putText(im0, "FPS: " + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                    im0 = cv2.resize(im0, (640,360))
                    cv2.imwrite(save_path, im0)
                    try :
                        cv2.imshow('iKurious Traffic Management', im0)
                        #print(txt_file_name+"_result.jpg")
                        cv2.imwrite(txt_file_name+"_result.jpg", im0)
                        if cv2.waitKey(1) % 256 == 27:  # ESC code
                            raise StopIteration
                    except KeyboardInterrupt:
                        raise StopIteration
                

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1000,700))
                vid_writer.write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    print("Total Sabes     : " + str(car_count))
    print("Total Batu belah: " + str(truck_count))
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

def connect(host='http://google.com'):
    try:
        urllib.request.urlopen(host) #Python 3.x
        return True
    except:
        return False


def reUPLOAD_img():
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
            if UploadIMGtoPedatiDouble(fileN,os.path.join(os.getcwd() + "\\images\\", fileN),fileN[-6],'re-upload'):
                print("Upload offline suksess...")
        else:
            if UploadIMGtoPedatiDouble(fileN,os.path.join(os.getcwd() + "/images/", fileN),fileN[-6],'re-upload'):
                print("Upload offline suksess...")

def get_date_time ():
    current_time = datetime.datetime.now()
    #print(current_time)
    tgl = str(current_time.year) + "-" + str(current_time.month) + "-" + str(current_time.day)
    #print(tgl)
    jam = str(current_time.hour) + ":" + str(current_time.minute) + ":" + str(current_time.second)
    #print(jam)
    str_date_time = str(current_time.year) + "_" + str(current_time.month) + "_" + str(current_time.day) +"_" + str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second)
    #print(str_date_time)
    return str_date_time

def delete_old_img():
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
def UploadIMGtoPedatiDouble(filenameSave, filename,filegambarstart,filenamestart, muatan, source):
    from datetime import datetime
    if connect():
        hostpedati = 'http://pedati.id:54100/mblb/api/main/insertcapturedouble'
        #hostpedati = 'http://pedati.id:54100/mblb/api/main/insertcapture'
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


def count_obj(box,w,h,id,direct,cls,im):
    global up_count,down_count,tracker1, tracker2, car_count, truck_count
    cx, cy = (int(box[0]+(box[2]-box[0])/2) , int(box[1]+(box[3]-box[1])/2))

    direc = os.getcwd()

    # For South
    if cy<= int(h//2):
        return

    if direct=="South":

        if cy > (lineDetect):
            if id not in tracker1:
                print(f"\nID: {id}, class: {cls}, H: {h} South\n")
                #print(h-300)     #660
                #label = f'{id} {names[c]} {conf:.2f}'
                #print(label)
                down_count +=1
                tracker1.append(id)
                #if cls==1:
                #    car_count+=1
                #elif cls==2:
                #    truck_count+=1
                if cls==1:
                    car_count+=1
                    print("Sabes")

                    #cv2.imwrite(direc+"\images\Sabes_"+str(car_count)+".jpg", im)

                    #str_date_time = get_date_time()

                    #filenameSave = "Img_" + str_date_time + "_0S.jpg"
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
                    img_resize = cv2.resize(im,(0,0),fx=0.5,fy=0.5)
                    cv2.imwrite(filename, img_resize)
                    cv2.imwrite(filenamestart, img_resize)
                    #UploadIMGtoPedatiDouble(filenameSave, filename, cls, 'live')
                    UploadIMGtoPedatiDouble(filenameSave, filename,filegambarstart, filenamestart, cls, 'live')
                elif cls==2:
                    truck_count+=1
                    print("Batu Belah")

                    #cv2.imwrite(direc+"\images\Batu_"+str(truck_count)+".jpg", im)
                    #str_date_time = get_date_time()
                    #print(str_date_time)

                    #filenameSave = "Img_" + str_date_time + "_0S.jpg"
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
                    img_resize = cv2.resize(im,(0,0),fx=0.5,fy=0.5)
                    cv2.imwrite(filename, img_resize)
                    cv2.imwrite(filenamestart, img_resize)
                    #UploadIMGtoPedatiDouble(filenameSave, filename, cls, 'live')
                    UploadIMGtoPedatiDouble(filenameSave, filename,filegambarstart, filenamestart, cls, 'live')


            
    elif direct=="North":
        if cy < (lineDetect):
            if id not in tracker2:
                print(f"\nID: {id}, H: {h} North\n")
                up_count +=1
                tracker2.append(id)
                
                if cls==2:
                    car_count+=1
                elif cls==7:
                    truck_count+=1


def direction(id,y):
    global dir_data

    if id not in dir_data:
        dir_data[id] = y
    else:
        diff = dir_data[id] -y

        if diff<0:
            return "South"
        else:
            return "North"


if __name__ == '__main__':
    __author__ = 'Mahimai Raja J'
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='galian_200epch_1k5.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='rtsp://admin:Kk123456@192.168.0.101/live', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[480], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--show-vid', default='store_true', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
