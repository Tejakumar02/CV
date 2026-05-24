# {0: 'gear_box', 1: 'engine_block', 2: 'aligner', 3: 'locking_lever',
#   4: 'tightening_tool'}
# 540, 960, 3



import time
import sys
from uuid import uuid4
import queue
import threading
import cv2
import math
from vidgear.gears import NetGear
from flask import Flask, Response, request, url_for
# from shapely.geometry import Polygon
from collections import deque
# from skimage.metrics import structural_similarity as ssim
# import pygame
import numpy as np
sys.path.append("C:/all_codes/yolov10")
from ultralytics import YOLO
import ultralytics
from ultralytics import YOLOv10
ultralytics.checks()


pos_model = YOLO('models/yolov8m-pose.onnx', task='pose',)

mcb_model = YOLOv10('models/gear_shop_best_25_7_24.onnx', task="detect",)
print(mcb_model.names)

insp_area = [210, 36, 594, 292]  # [463, 176, 790, 310]
#blank_ins_pic = cv2.imread("blank_inspection_area3.png")
#blank_ins_pic_ssim = cv2.cvtColor(
#    blank_ins_pic[insp_area[1]:insp_area[3], insp_area[0]:insp_area[2]], cv2.COLOR_BGR2GRAY)
all_steps = ['Place Engine + GB','Locate Locking Lever','Place Aligner','Remove Aligner',
             'Couple Engine + GB','Tighten Bolts','Remove Locking Lever']
cycle_check = []
cycle_check_time = []

loc = threading.Lock()  # Semaphore()
dummy_img = cv2.imread("defect_scanner_logo.png")
logo = cv2.imread("defect-scanner-logo-transparent-cropped.png")
frame_ori = dummy_img.copy()
rec_vid = False
video_rec = 0
options = {"max_retries": 130, "request_timeout": 20, }
# client = NetGear(receive_mode=True, address="0.0.0.0", port="12345",**options,)
cycle_check_f = 0
cycle_original_sequence = [0, 4, 1, 5, 2, 3]
without_detection_timer = int(time.time())
every_sequence_timer = int(time.time())
cycle_start_time = None
sav_lab = 0
screw_drill_pos=[]
all_cls = []
cycle_reset_f=0
cycle_break = 0
conf_lev=3
cls_conf = deque(maxlen=15)
show_blink_rect = True
tightening_toolf = 0
eng_cxy = []
gb_cxy = []
locking_leverf = 0


app = Flask(__name__,)


class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)


@app.route("/favicon.ico")
def favicon():
    return url_for('static', filename='data:,')


@app.route('/')
def index():
    print("here")
    return "Ashok Leyland"


@app.route('/savl')
def sav_label():
    global sav_lab
    sav_lab = 1-sav_lab
    return "Label save"


@app.route('/clear')
def cls():
    global cycle_check
    global cycle_check_f
    cycle_check.clear()
    cycle_check_f = 0
    return "Done"


@app.route('/record')
def record():
    global rec_vid
    global video_rec
    if rec_vid:
        video_rec.release()
        rec_vid = False
        return "Recording stoped"
    elif rec_vid == False:
        video_rec = cv2.VideoWriter(f"D:/recorded_videos/{str(uuid4())}.avi",
                                    cv2.VideoWriter_fourcc(*'XVID'),
                                    6, (960, 540))
        rec_vid = True
        return "Recording start"
    return "Some error in recording"


# def calculate_iou(box_1, box_2):
#     poly_1 = Polygon(box_1)
#     poly_2 = Polygon(box_2)
#     iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
#     return iou

def calculate_iou(rect1, rect2):
    x1_rect1, y1_rect1, x2_rect1, y2_rect1 = rect1
    x1_rect2, y1_rect2, x2_rect2, y2_rect2 = rect2

    # Calculate the (x, y)-coordinates of the intersection rectangle
    x_left = max(x1_rect1, x1_rect2)
    y_top = max(y1_rect1, y1_rect2)
    x_right = min(x2_rect1, x2_rect2)
    y_bottom = min(y2_rect1, y2_rect2)

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of the intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the area of both rectangles
    rect1_area = (x2_rect1 - x1_rect1) * (y2_rect1 - y1_rect1)
    rect2_area = (x2_rect2 - x1_rect2) * (y2_rect2 - y1_rect2)

    # Calculate the union area
    union_area = rect1_area + rect2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou

def reset_blank_desk(frame):
    global insp_area
    global blank_ins_pic_ssim
    print("Reseted blank desk ##############################")
    blank_ins_pic_ssim = cv2.cvtColor(
        frame[insp_area[1]:insp_area[3], insp_area[0]:insp_area[2]], cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("./crop_a.png", blank_ins_pic_ssim)

def pyg(frame, details_=[], time_=[]):
    global all_steps
    global cycle_check_f
    global cycle_start_time
    global all_cls
    global cycle_reset_f
    global cycle_break
    global cls_conf
    global conf_lev
    global show_blink_rect
    global locking_leverf
    # global hand_s
    # global hand_e
    elapsed_time_str = 0
    if cycle_start_time is not None:
        elapsed_time = time.time() - cycle_start_time
        elapsed_time_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
    if len(details_) == 0:
        return frame
    details = [0]
    for i in details_:
        details.append(all_steps[i])
    cycle_error = 0
    s_x = 600
    s_y = 0
    yellow = [14, 201, 255]
    green = [0, 255, 0]
    red = [0,0,255]

    overlay = frame.copy()
    cv2.rectangle(overlay, (s_x, s_y), (s_x +
                  360, s_y + 40*len(details)), [0, 0, 0], -1)

    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # back_img = #np.zeros((40*len(details), 320, 3), dtype=np.uint8)
    # frame[s_y:s_y+back_img.shape[0], s_x:s_x+back_img.shape[1]] = back_img
    cv2.rectangle(frame, (s_x, s_y),
                  (s_x+320, 40*len(details)), [255, 0, 0], 2)

    for i, val in enumerate(details):
        if i == 0:
            # if cycle_check_f == 0:
            t = f"Cycle Status: In Progress"# {elapsed_time_str} Sec"
            c_t = [14, 201, 255]
            # else:
            #     t = "Cycle Status: Sequence Break"
            #     c_t = [0, 0, 255]
            if len(details_)==7 and cls_conf.count(3)==0 and locking_leverf==0:
                t = f"Cycle Status: Completed"# {elapsed_time_str} Sec"
                c_t = [0, 255, 0]
            elif (2<len(details_)<7 and cls_conf.count(3)==0 and cls_conf.count(0)>=conf_lev and cls_conf.count(1)>=conf_lev) or cycle_reset_f>=2 or cycle_break==1:
                t = "Cycle Status: Sequence Break"
                c_t = [0, 0, 255]
                cycle_error = 1
                print("Error: ",(1<len(details_)<7 and 3 not in all_cls and 0 in all_cls and 1 in all_cls),cycle_reset_f>=2,cycle_break==1)

            cv2.putText(frame, t, (s_x+10, 25),
                        cv2.FONT_ITALIC, 0.6, c_t, 1)
        elif i < len(details)-1:
            cv2.putText(frame, str(i), (s_x+15, 40*i+25),
                        cv2.FONT_ITALIC, 0.6, green, 2)

            cv2.putText(frame, val, (s_x+55, 40*i+25),
                        cv2.FONT_ITALIC, 0.6, green, 1)

            cv2.circle(frame, (900, 40*i+20), 10, green, -1)

            # cv2.putText(frame, "{:.1f}".format(time_[i]-time_[i-1]), (s_x+325, 40*i+25),
                        # cv2.FONT_ITALIC, 0.6, green, 1)
        else:
            if len(details_)==7 and cls_conf.count(3)==0:
                col=green
                # cv2.putText(frame, "{:.1f}".format(time.time()-time_[i-1]), (s_x+325, 40*i+25),
                #             cv2.FONT_ITALIC, 0.6, green, 1)
            
            elif cycle_error ==1 :
                show_blink_rect = not show_blink_rect
                col=[255,255,255]
                # ori_img = frame.copy()
                if show_blink_rect:
                    overlay_ = frame.copy()
                    cv2.rectangle(overlay_,(s_x, 40*i), (s_x+320, 40*i+40),red,-1)
                    cv2.addWeighted(overlay_, 0.6, frame, 0.4, 0, frame)
            else:
                col=yellow
            cv2.putText(frame, str(i), (s_x+15, 40*i+25),
                        cv2.FONT_ITALIC, 0.6, col, 2)

            cv2.putText(frame, val, (s_x+55, 40*i+25),
                        cv2.FONT_ITALIC, 0.6, col, 1)

            cv2.circle(frame, (900, 40*i+20), 10, col, -1)
        cv2.line(frame, (s_x, 40*i), (s_x+320, 40*i), [255, 0, 0], 2)

    cv2.line(frame, (s_x+40, 40), (s_x+40, 40*len(details)), [255, 0, 0], 2)
    cv2.line(frame, (s_x+280, 40), (s_x+280, 40*len(details)), [255, 0, 0], 2)
    return frame


def frameInf(frame, stime):
    global dummy_img
    global logo
    global pos_model
    global all_steps
    global cycle_check
    global cycle_check_time
    global insp_area
    global blank_ins_pic
    global cycle_check_f
    global cycle_original_sequence
    global without_detection_timer
    global every_sequence_timer
    global sav_lab
    global cycle_start_time
    global screw_drill_pos
    global rec_vid
    global video_rec
    global all_cls
    global cycle_reset_f
    global cycle_break
    global cls_conf
    global conf_lev
    global tightening_toolf
    global eng_cxy
    global gb_cxy
    global locking_leverf
    # frame = cv2.resize(frame, (640, 360))
    if frame is None:
        return (dummy_img)
    
    mcb_results = mcb_model.predict(
            frame, conf=0.4, iou=0.8, imgsz=640, verbose=False)
    # frame = mcb_results[0].plot()#conf=False,labels=False)
    
    results = pos_model.predict(
        frame, conf=0.5, iou=0.6, imgsz=640, verbose=False)
    frame = results[0].plot(conf=False,labels=False,boxes=False)
    
    #print(mcb_results[0].boxes.xyxy.tolist())
    all_cls = mcb_results[0].boxes.cls.tolist()
    all_cls = list(map(int, all_cls))
    if len(all_cls)>0:
        for i in all_cls:
            cls_conf.append(i)
    else:
        cls_conf.append(10)
    print("all cls",all_cls)
    if cls_conf.count(0)>=conf_lev or cls_conf.count(1)>=conf_lev:
        if 0 not in cycle_check:
            cycle_check.append(0)  
        cycle_reset_f = 0
        # if 3 in all_cls:
        #     if 1 not in cycle_check:
        #         cycle_check.append(1)
        #     if 
    else:
        cycle_reset_f+=1
        if cycle_reset_f==5:
            cycle_check.clear()
            tightening_toolf = 0
            eng_cxy.clear()
            locking_leverf = 0
    if len(cycle_check)>=3 and cls_conf.count(2)>=conf_lev and (cls_conf.count(4)>=conf_lev or cls_conf.count(5)>=conf_lev):
        try:
            cycle_check.remove(4)
            cycle_check.remove(5)
            cycle_check.remove(6)
        except:
            pass
    if 1 not in cycle_check and cls_conf.count(0)>=conf_lev and cls_conf.count(1)>=conf_lev:
        cycle_check.append(1)
    if 2 not in cycle_check and len(cycle_check)>=2 and cls_conf.count(3)>=conf_lev: 
        cycle_check.append(2)
    if 3 not in cycle_check and len(cycle_check)>=3 and cls_conf.count(2)>=conf_lev and cls_conf.count(3)>=conf_lev: 
        cycle_check.append(3)
    if 4 not in cycle_check and len(cycle_check)>=4 and cls_conf.count(2)==0 and cls_conf.count(3)>=conf_lev: 
        cycle_check.append(4)
    if len(cycle_check)>=5 and cls_conf.count(3)>=conf_lev:# or (0 in all_cls and 1 in all_cls):
        try:
            all_xy=mcb_results[0].boxes.xyxy.tolist()
            eng_xywh=all_xy[all_cls.index(1)]
            gb_xywh=all_xy[all_cls.index(0)]
            iou_=gb_xywh[0]-eng_xywh[2]#calculate_iou(eng_xywh,gb_xywh)
            engx_c,engy_c = int(eng_xywh[0]+abs(eng_xywh[2]-eng_xywh[0])/2),int(eng_xywh[1]+abs(eng_xywh[3]-eng_xywh[1])/2)
            gbx_c,gby_c = int(gb_xywh[0]+abs(gb_xywh[2]-gb_xywh[0])/2),int(gb_xywh[1]+abs(gb_xywh[3]-gb_xywh[1])/2)
            cv2.circle(frame,(engx_c,engy_c),3,[0,0,255],-1)
            if len(eng_cxy)<5:
                eng_cxy.append([engx_c,engy_c])
                gb_cxy.append([gbx_c,gby_c])
            else:
                if len(eng_cxy)==5: # avg of center 
                    eng_cxy.append([(eng_cxy[0][0]+eng_cxy[1][0]+eng_cxy[2][0]+eng_cxy[3][0])/4,(eng_cxy[0][1]+eng_cxy[1][1]+eng_cxy[2][1]+eng_cxy[3][1])/4])
                    gb_cxy.append([(gb_cxy[0][0]+gb_cxy[1][0]+gb_cxy[2][0]+gb_cxy[3][0])/4,(gb_cxy[0][1]+gb_cxy[1][1]+gb_cxy[2][1]+gb_cxy[3][1])/4])
                    
                print("Distance : ****************** ",math.dist([engx_c,engy_c],eng_cxy[-1]),math.dist([gbx_c,gby_c],gb_cxy[-1]))
            print(iou_,gb_xywh[0]-eng_xywh[2])
            if iou_<1.2 and 5 not in cycle_check:
                cycle_check.append(5)
            elif iou_>2 and 5 in cycle_check:
                try:
                    cycle_check.remove(5)
                    cycle_check.remove(6)
                except:
                    pass
        except:
            pass
    # if tightening_toolf==0:
    if 6 not in cycle_check and 7>len(cycle_check)>=6 and cls_conf.count(4)==0 and cls_conf.count(3)>=conf_lev:
        cycle_check.append(6)
        # tightening_toolf=1
    # else:
    #     if 6 not in cycle_check and 7>len(cycle_check)>=6 and cls_conf.count(4)>=conf_lev and cls_conf.count(3)>=conf_lev:
    #         cycle_check.append(6)
    
    


    cycle_break = 0
    # All types of error 
    if cls_conf.count(0)>=conf_lev and cls_conf.count(1)>=conf_lev:
        # aligner presence before locking lever
        if 2==len(cycle_check) and cls_conf.count(2)>=conf_lev and cycle_break==0:
            cycle_break=1
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # aligner not used before couple
        if 2<=len(cycle_check)<=3 and cycle_break==0 :
            try:
                all_xy=mcb_results[0].boxes.xyxy.tolist()
                eng_xywh=all_xy[all_cls.index(1)]
                gb_xywh=all_xy[all_cls.index(0)]
                iou_=gb_xywh[0]-eng_xywh[2]#calculate_iou(eng_xywh,gb_xywh)
                print(iou_,gb_xywh[0]-eng_xywh[2])
                if iou_<1.2 and cls_conf.count(5)==0:
                    cycle_break = 1
                    print("#############################################")
            except:
                pass
    
    if len(eng_cxy)>5:
        try:
            all_xy=mcb_results[0].boxes.xyxy.tolist()
            eng_xywh=all_xy[all_cls.index(1)]
            engx_c,engy_c = int(eng_xywh[0]+abs(eng_xywh[2]-eng_xywh[0])/2),int(eng_xywh[1]+abs(eng_xywh[3]-eng_xywh[1])/2)
            gbx_c,gby_c = int(gb_xywh[0]+abs(gb_xywh[2]-gb_xywh[0])/2),int(gb_xywh[1]+abs(gb_xywh[3]-gb_xywh[1])/2)
            print("Distance : ****************** ",math.dist([engx_c,engy_c],eng_cxy[-1]),math.dist([gbx_c,gby_c],gb_cxy[-1]))
            if math.dist([engx_c,engy_c],eng_cxy[-1])>7 and cls_conf.count(3)>0:
                cycle_break=1
                locking_leverf = 1
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
        except:
            pass

    
    
    
    # temp_list2 = mcb_results[0].boxes.xyxy.tolist()
    # if (temp_list.count(0)==5 or temp_list.count(2)==5) and cycle_check_f<5:
    #     print(temp_list)
    #     print(temp_list2)
    #     cycle_check_f+=1
    #     if cycle_check_f==5:
    #         temp_list2.sort()
    #         calibrate_posi(temp_list2)
    # # elif cycle_check_f>=5:
    # #     for i in screw_drill_pos:
    # #         cv2.circle(frame,(i[0],i[1]),10,[255,0,0],2)
    # if temp_list.count(0)==5 and len(cycle_check)>0:
    #     cycle_check.clear()
    #     cycle_check_f=7
    # if (temp_list.count(2)==5) and len(cycle_check)==0:
    #     cycle_check.append(0)

    # if 1 in temp_list:
    #     x1,y1,x2,y2 = temp_list2[temp_list.index(1)]
    #     x_c,y_c = int(x1+abs(x2-x1)/2),int(y1+abs(y2-y1)/2)
    #     # cv2.circle(frame,(x_c,y_c),3,[0,0,255],-1)
    #     #print(x_c,y_c)

    #     for z,i in enumerate(screw_drill_pos):
    #         dis = math.sqrt((x_c-i[0])**2+(y_c-i[1])**2)
    #         # print("dis",dis)
    #         if dis<15:
    #             # cv2.circle(frame,(x_c,y_c),4,[0,255,0],-1)
    #             print(z)
    #             if (z+1) not in cycle_check:
    #                 print("************************************screw tightened***************************")
    #                 cycle_check.append(z+1)
    #     if cycle_original_sequence[0:len(cycle_check)] != cycle_check:
    #         print("Sequence breaked")
    # elif len(cycle_check)==6:
    #     cycle_check_f=10
    #print(mcb_results[0].boxes.cls.tolist())
    # frame = mcb_results[0].plot()#conf=False,labels=False)
    frame = cv2.resize(frame,(960,540))
    frame[510:510+logo.shape[0], 5:5+logo.shape[1]] = logo
    cycle_check.sort()
    # print("queue: ",cls_conf)
    # print("cycle check: ",cycle_check)
    frame=pyg(frame,cycle_check)
    if rec_vid:
        video_rec.write(frame)

    return frame
    results = pos_model.predict(
        frame, conf=0.5, iou=0.6, imgsz=640, verbose=False)

    if sav_lab == 0:
        frame = results[0].plot(conf=False)  # conf=False
        mcb_results = mcb_model.predict(
            frame, conf=0.4, iou=0.8, imgsz=960, verbose=False)
    elif sav_lab == 1:
        im_name = "C:/all_codes/dataset_mcb/"+str(uuid4())+".png"
        cv2.imwrite(im_name, frame)
        mcb_results = mcb_model.predict(
            im_name, conf=0.4, iou=0.8, imgsz=960,  save_txt=True, project="C:/all_codes", name="dataset_mcb/", exist_ok=True, verbose=False)

    # mcb_det_area = mcb_results[0].boxes.xyxy.tolist()
    # reset cycle
    ssim_v = ssim(blank_ins_pic_ssim, cv2.cvtColor(
        frame[insp_area[1]:insp_area[3], insp_area[0]:insp_area[2]], cv2.COLOR_BGR2GRAY))
    # print("ssim re ", temp)
    if ssim_v > 0.92:  # and int(time.time())-every_sequence_timer>10:
        cycle_check.clear()
        cycle_check_time.clear()
        cycle_check_f = 0
        cycle_start_time = time.time()
        every_sequence_timer = int(time.time())
    frame = mcb_results[0].plot(conf=False)
    # frame[610:610+logo.shape[0], 5:5+logo.shape[1]] = logo
    # print("result",stime,time.time(),time.time()-stime)
    # print("keypoints: ", results[0].keypoints.xy.tolist()[0][9:11])
    hand_points = results[0].keypoints.xy.tolist()[0][9:11]
    # [507, 330, 708, 400]

    # for i in hand_points:
    #     cv2.circle(frame, (int(i[0]),int(i[1])), 5, [0,255,0], -1)

    if len(hand_points) > 0:
        # if (507 < hand_points[1][0] < 708) and (330 < hand_points[1][1]<400):
        if (350 < hand_points[1][0] < 575) and (330 < hand_points[1][1] < 390):
            cycle_check.clear()
            cycle_check_time.clear()
            cycle_check.append(0)
            cycle_check_time.append(time.time())
            cycle_start_time = time.time()
            print("&&&&&&&&&&&&&&&&based on hand position cycle START &&&&&&&&&&&&&")
        elif (55 < hand_points[0][0] < 292) and (330 < hand_points[0][1] < 390):
            cycle_check.clear()
            cycle_check_time.clear()
            print("&&&&&&&&&&&&&&&&based on hand position cycle END &&&&&&&&&&&&&")

    mcb_cls_re = mcb_results[0].boxes.cls.tolist()
    mcb_cls_re = list(map(int, mcb_cls_re))
    # for i, value in enumerate(mcb_det_area):
    #     iou_ = calculate_iou([[insp_area[0], insp_area[1]], [insp_area[2], insp_area[1]], [
    #                          insp_area[2], insp_area[3]], [insp_area[0], insp_area[3]]], [[value[0], value[1]], [value[2], value[1]], [
    #                              value[2], value[3]], [value[0], value[3]]])
    #     print(iou_)

    # add new detected face
    if cycle_check_f == 0 and len(mcb_cls_re) > 0:
        if mcb_cls_re[0] != 0:
            mcb_cls_re.insert(0, 0)
        for diff in list(set(mcb_cls_re)-set(cycle_check)):
            if diff not in [7, 8]:
                cycle_check.append(diff)
                cycle_check_time.append(time.time())
            every_sequence_timer = int(time.time())
        # if len(cycle_check) > 0:
        #     if cycle_original_sequence[0:len(cycle_check)] != cycle_check:
        #         print("Wrong sequence $$$$$$$$$$$$$$$$$")
        #         cycle_check_f = 1

    if len(mcb_cls_re) == 0:   # ssim image update ssim chenck <
        if int(time.time())-without_detection_timer > 40 and ssim_v <= 0.93:
            reset_blank_desk(frame)
            without_detection_timer = int(time.time())
    else:
        without_detection_timer = int(time.time())

    # int(time.time())-without_detection_timer
    print("ssim re ", ssim_v, cycle_check_time, sav_lab)
    # for i, value in enumerate(all_steps):
    #     if i in cycle_check:
    #         color_ = [0, 255, 0]
    #     else:
    #         color_ = [0, 0, 255]
    #     cv2.putText(frame, value, (10, 60+(30*i)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_, 2)
    # cv2.rectangle(frame,(insp_area[0],insp_area[1]),(insp_area[2],insp_area[3]),[0,0,255],1)
    print(mcb_cls_re, cycle_check)

    # if all face detected
    if [1, 2, 3, 4, 5, 6] not in mcb_cls_re and len(cycle_check) == 7:
        cycle_check.clear()
        cycle_start_time = time.time()

    frame[510:510+logo.shape[0], 5:5+logo.shape[1]] = logo
    # try:
    #     cycle_check.remove(7)
    #     cycle_check.remove(8)
    # except:
    #     pass
    return (pyg(frame, cycle_check, cycle_check_time))
    # return frame


@app.route('/img_update', methods=['GET'])
def update():
    global frame_queue
    global client
    global frame_ori
    global rec_vid
    global video_rec
    global screen
    if request.method == 'GET':
        # if client != 0:
        #     client.close()
        client = NetGear(receive_mode=True, address="0.0.0.0",
                         port="12345", protocol="tcp", pattern=1, **options,)
        # yield "200"
    while True:
        try:
            frame = client.recv()
            if frame is None:
                print("Client closed by None")
                client.close()
                break
            # print(time.time())
            # continue
            else:
                frame_ori = frame.copy()
                # if not frame_queue.full():
                t = ThreadWithResult(
                    target=frameInf, args=(frame, time.time()))
                t.start()
                frame_queue.put(t)
                # if rec_vid:
                #     video_rec.write(frame)
        except Exception as e:
            print("Client closed by Exception", e)
            client.close()
            video_rec.release()
            rec_vid = False
            break

    return "200"


flag = 1
client = 0
th_l = 0
_, buffer = cv2.imencode('.jpg', dummy_img)
frame_queue = queue.Queue(90)


def generate_frames():
    global buffer
    global frame_queue
    global th_l
    while True:
        if th_l != 0:
            # print("th_l=1",end=" ")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        loc.acquire()
        th_l = 1
        if not frame_queue.empty():
            temp = frame_queue.get()
            temp.join()
            try:
                _, buffer = cv2.imencode('.jpg', temp.result)
            except Exception as e:
                print(e)
                pass
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        th_l = 0
        loc.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    app.run(host='127.0.0.1', port=5000)
