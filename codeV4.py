# names:
#   0: end_cover
#   1: end_cover_bolt
#   2: tower
#   3: clamp
#   4: hammer
#   5: lock_pin_interlock
#   6: circlip_interlock
#   7: drain_plug
#   8: loctite
#   9: cutoff_valve
#   10: tower_movement
#   11: drill_bit1
#   12: drill_bit2
#   13: fixture
#   14: drillm
#   15: hand_gloves
# 30: end cover tightening 
# 31: Drain plug tightening
# 32: Cut-off valve tightening

import time
import sys
from uuid import uuid4
import queue
import threading
import cv2
import math
from collections import deque
# from vidgear.gears import NetGear
from flask import Flask, Response, request, url_for
# from shapely.geometry import Polygon
# from skimage.metrics import structural_similarity as ssim
# import pygame
import numpy as np
# sys.path.append("C:/AL_chennai_tower/yolov10")
from ultralytics import YOLO
import ultralytics
# from ultralytics import YOLOv10
ultralytics.checks()


pos_model = YOLO('models/yolov8n-pose.pt', task='pose',)
pos_model.predict(
            "defect_scanner_logo.png", conf=0.4, iou=0.8, imgsz=640, verbose=False,device =[0])
tower_model = YOLO('models/AL_tower_081124.pt', task="detect")
print(tower_model.names)
tower_model.predict(
            "defect_scanner_logo.png", conf=0.4, iou=0.8, imgsz=640, verbose=False,device =[0])

insp_area = [210, 36, 594, 292]  # [463, 176, 790, 310]
#blank_ins_pic = cv2.imread("blank_inspection_area3.png")
#blank_ins_pic_ssim = cv2.cvtColor(
#    blank_ins_pic[insp_area[1]:insp_area[3], insp_area[0]:insp_area[2]], cv2.COLOR_BGR2GRAY)
all_steps = ["End cover fitment","End cover bolt fitment","Clamping","Interlock pin fitment","Hammering Interlock","Circlip fitment",
             "Locking the circlip","Drain plug fitment","Apply Loctite ","Cut-off valve fitment",
             "End cover tightening","Drain plug tightening","Cutoff valve tightening","Detent Plunger","Yellow Spring","Pink Spring"]
cycle_check = []
cycle_check_time = []

loc = threading.Lock()  # Semaphore()
dummy_img = cv2.imread("defect_scanner_logo.png")
logo = cv2.imread("defect-scanner-logo-transparent-cropped.png")
frame_ori = dummy_img.copy()
rec_vid = False
video_rec = 0
options = {"max_retries": 2, "request_timeout": 5, }
# client = NetGear(receive_mode=True, address="0.0.0.0", port="12345",**options,)
cycle_check_f = 0
cycle_original_sequence = [0, 4, 1, 5, 2, 3]
without_detection_timer = int(time.time())
every_sequence_timer = int(time.time())
cycle_start_time = None
# sav_lab = 0
screw_drill_pos=[]
cls_ins = deque(maxlen=30)
tower_conf = deque(maxlen=20)
conf_lev = 3
trigger_one_flag = 0
sequence_break_f = 0
show_blink_rect = 0
time_c = 0
detent_box_pos = [41,232,142,309]
yellowS_box_pos = [216,110,279,168]
pinkS_box_pos = [205,187,271,238]
hand_f = 0
spring_plungerf = 0


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


# @app.route('/savl')
# def sav_label():
#     global sav_lab
#     sav_lab = 1-sav_lab
#     return "Label save"


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
        video_rec = cv2.VideoWriter(f"D:/recorded_video_al_tower/{str(uuid4())}.mp4",
                                    cv2.VideoWriter_fourcc(*'H264'),
                                    8, (960, 540))
        rec_vid = True
        return "Recording start"
    return "Some error in recording"


# def calculate_iou(box_1, box_2):
#     poly_1 = Polygon(box_1)
#     poly_2 = Polygon(box_2)
#     iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
#     return iou




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
    global sequence_break_f
    global show_blink_rect
    global spring_plungerf
    # global hand_s
    # global hand_e
    elapsed_time_str = 0
    if cycle_start_time is not None:
        elapsed_time = time.time() - cycle_start_time
        elapsed_time_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
    sequence_break_f_ = sequence_break_f
    if len(details_) == 0:
        return frame
    elif len(details_)>12:
        if sequence_break_f!=0:
            print("Sequence break: ", sequence_break_f,sequence_break_f_,len(details_),
                sequence_break_f- (len(details_)-12))
            sequence_break_f_ =sequence_break_f- (len(details_)-12)
        details_ = details_[-12:]
        
    details = [0]
    for i in details_:
        details.append(all_steps[i])
    s_x = 600
    s_y = 0
    yellow = [14, 201, 255]
    green = [0, 255, 0]
    white = [255,255,255]
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
            if cycle_check_f==10:
                t = f"Cycle Status: Completed"# {elapsed_time_str} Sec"
                c_t = [0, 255, 0]
            if sequence_break_f_!=0:
                t = "Cycle Status: Sequence Break"
                c_t = red
            cv2.putText(frame, t, (s_x+10, 25),
                        cv2.FONT_ITALIC, 0.6, c_t, 1)
        elif i < len(details)-1 and (0==sequence_break_f_ or i<sequence_break_f_):
            cv2.putText(frame, str(i), (s_x+15, 40*i+25),
                        cv2.FONT_ITALIC, 0.6, green, 2)

            cv2.putText(frame, val, (s_x+55, 40*i+25),
                        cv2.FONT_ITALIC, 0.6, green, 1)

            cv2.circle(frame, (900, 40*i+20), 10, green, -1)

            # cv2.putText(frame, "{:.1f}".format(time_[i]-time_[i-1]), (s_x+325, 40*i+25),
                        # cv2.FONT_ITALIC, 0.6, green, 1)
        else:
            if cycle_check_f==10:
                cv2.putText(frame, str(i), (s_x+15, 40*i+25),
                            cv2.FONT_ITALIC, 0.6, green, 2)

                cv2.putText(frame, val, (s_x+55, 40*i+25),
                            cv2.FONT_ITALIC, 0.6, green, 1)

                cv2.circle(frame, (900, 40*i+20), 10, green, -1)

                # cv2.putText(frame, "{:.1f}".format(time.time()-time_[i-1]), (s_x+325, 40*i+25),
                #             cv2.FONT_ITALIC, 0.6, green, 1)
            elif sequence_break_f_!=0:
                if i < len(details)-1:
                    cv2.putText(frame, str(i), (s_x+15, 40*i+25),
                                cv2.FONT_ITALIC, 0.6, red, 2)

                    cv2.putText(frame, val, (s_x+55, 40*i+25),
                                cv2.FONT_ITALIC, 0.6, red, 1)

                    cv2.circle(frame, (900, 40*i+20), 10, red, -1)
                else:
                    show_blink_rect = not show_blink_rect
                    # ori_img = frame.copy()
                    if show_blink_rect:
                        overlay_ = frame.copy()
                        cv2.rectangle(overlay_,(s_x, 40*i), (s_x+320, 40*i+40),red,-1)
                        cv2.addWeighted(overlay_, 0.6, frame, 0.4, 0, frame)
                    cv2.putText(frame, str(i), (s_x+15, 40*i+25),
                                cv2.FONT_ITALIC, 0.6, white, 2)

                    cv2.putText(frame, val, (s_x+55, 40*i+25),
                                cv2.FONT_ITALIC, 0.6, white, 1)

                    cv2.circle(frame, (900, 40*i+20), 10, white, -1)

            else:
                if details_[-1] in [13,14,15] and spring_plungerf == 1:
                    cv2.putText(frame, str(i), (s_x+15, 40*i+25),
                            cv2.FONT_ITALIC, 0.6, green, 2)
                    cv2.putText(frame, val, (s_x+55, 40*i+25),
                                cv2.FONT_ITALIC, 0.6, green, 1)

                    cv2.circle(frame, (900, 40*i+20), 10, green, -1)
                else:
                    cv2.putText(frame, str(i), (s_x+15, 40*i+25),
                                cv2.FONT_ITALIC, 0.6, yellow, 2)

                    cv2.putText(frame, val, (s_x+55, 40*i+25),
                                cv2.FONT_ITALIC, 0.6, yellow, 1)

                    cv2.circle(frame, (900, 40*i+20), 10, yellow, -1)
        cv2.line(frame, (s_x, 40*i), (s_x+320, 40*i), [255, 0, 0], 2)

    cv2.line(frame, (s_x+40, 40), (s_x+40, 40*len(details)), [255, 0, 0], 2)
    cv2.line(frame, (s_x+280, 40), (s_x+280, 40*len(details)), [255, 0, 0], 2)
    return frame


def frameInf(frame, stime):
    global dummy_img
    global logo
    global all_steps
    global cycle_check
    global cycle_check_time
    global insp_area
    global cycle_check_f
    global cycle_original_sequence
    global without_detection_timer
    global every_sequence_timer
    # global sav_lab
    global cycle_start_time
    global screw_drill_pos
    global rec_vid
    global video_rec
    global tower_model, pos_model
    global cls_ins, conf_lev, tower_conf
    global trigger_one_flag
    global sequence_break_f
    global time_c, spring_plungerf
    global detent_box_pos, pinkS_box_pos, yellowS_box_pos, hand_f
    # frame = cv2.resize(frame, (640, 360))
    # frame = cv2.resize(frame,(530,496))
    # print(time.time())
    s=time.time()
    if frame is None:
        return (dummy_img)
    tower_results = tower_model.predict(
            frame, conf=0.4, iou=0.5, imgsz=640, verbose=False,device =[0])
    pos_results = pos_model.predict(
            frame, conf=0.4, iou=0.5, imgsz=640, verbose=False,device =0)
    temp_list = tower_results[0].boxes.cls.tolist()
    # print("temp list:",set(temp_list),time.time()-stime)
    temp_xyxy = tower_results[0].boxes.xyxy.tolist()
    tower_detf = 0
    if len(temp_list)>0: # and len(cls_ins)>0:
        for i in set(temp_list):
            if i == 2:
                tower_conf.append(2)
                tower_detf=1
            else:
                cls_ins.append(i)
    elif len(cls_ins)>0:
        cls_ins.append(50)
    if tower_detf==0:
        tower_conf.append(10)
    
    # Tower presence check
    if tower_conf.count(2)>=3 or len(cycle_check)>2:
        # End cover
        if 0 not in cycle_check and cls_ins.count(0)>=3:
            cycle_check.append(0)
        # End cover bolt
        if 1 not in cycle_check and 0 in cycle_check and cls_ins.count(1)>=3:
            cycle_check.append(1)
        # Clamp 
        if 2 not in cycle_check and cls_ins.count(3)>=3:
            cycle_check.append(2)
        # Interlock lock pin
        if 3 not in cycle_check and cls_ins.count(5)>=3:
            cycle_check.append(3)
            time_c =time.time()
        # Hammering
        if 4 not in cycle_check and 3 in cycle_check and cls_ins.count(4)>=2 and cycle_check[-1]==3:
            cycle_check.append(4)
        elif cls_ins.count(6)>=3 and 4 not in cycle_check and sequence_break_f==0:
            sequence_break_f = len(cycle_check)
        # Circlip interlock pin
        if 5 not in cycle_check and 4 in cycle_check and 3 in cycle_check and cls_ins.count(6)>=3 and time.time()-time_c>30:
            cycle_check.append(5)
        # Hammering
        if 6 not in cycle_check and 5 in cycle_check and cls_ins.count(4)>=2 and cycle_check[-1]==5:
            cycle_check.append(6)
        # Drain plug
        if 7 not in cycle_check and cls_ins.count(7)>=3:
            cycle_check.append(7)
        # Apply loctite
        if 8 not in cycle_check and 7 in cycle_check and cls_ins.count(8)>=3:
            cycle_check.append(8)
        # Cut-off valve
        if 9 not in cycle_check and cls_ins.count(9)>=3:
            cycle_check.append(9) 
        # End cover tightening
        if 10 not in cycle_check and cls_ins.count(30)>=3:
            cycle_check.append(10)
        # Drain plug tightening
        if 11 not in cycle_check and cls_ins.count(31)>=3:
            cycle_check.append(11)
        # Cut-off valve tightening
        if 12 not in cycle_check and cls_ins.count(32)>=3:
            cycle_check.append(12)
        # Drill machine pose
        if 14 in temp_list:
            drill_xyxy = temp_xyxy[temp_list.index(14)]
            drillx_c,drilly_c = int(drill_xyxy[0]+abs(drill_xyxy[2]-drill_xyxy[0])/2),int(drill_xyxy[1]+abs(drill_xyxy[3]-drill_xyxy[1])/2)
            print("drill pose: ",drillx_c,drilly_c, cycle_check)
            if 10 not in cycle_check:
                if 230<drillx_c<265 and 310<drilly_c<330:
                    #End cover drill
                    cv2.putText(frame, 'End cover 1st', (20, 20),
                                cv2.FONT_ITALIC, 0.6, [0,0,255], 1)
                    cls_ins.append(30)
                elif 280<drillx_c<325 and 320<drilly_c<345:
                    #End cover drill
                    cv2.putText(frame, 'End cover 2nd', (20, 40),
                                cv2.FONT_ITALIC, 0.6, [0,0,255], 1)
                    cls_ins.append(30)
            
            if 12 not in cycle_check:
                if 120<drillx_c<150 and (0<drilly_c<18 or 25<drilly_c<40):
                    #Cutoff valve drill
                    cv2.putText(frame, 'Cutoff valve 1st', (20, 20),
                                cv2.FONT_ITALIC, 0.6, [0,0,255], 1)
                    cls_ins.append(32)
                elif 120<drillx_c<150 and (0<drilly_c<18 or 45<drilly_c<55):
                    #Cutoff valve drill
                    cv2.putText(frame, 'Cutoff valve 2nd', (20, 50),
                                cv2.FONT_ITALIC, 0.6, [0,0,255], 1)
                    cls_ins.append(32)
            if 11 not in cycle_check:
                if 415<drillx_c<438 and (145<drilly_c<157 or 113<drilly_c<125):
                    #Drain plug drill
                    cv2.putText(frame, 'Drain plug', (20, 50),
                                cv2.FONT_ITALIC, 0.6, [0,255,0], 1)
                    cls_ins.append(31)
                elif 140<drillx_c<165 and (68<drilly_c<88 or 96<drilly_c<116):
                    #Drain plug drill
                    cv2.putText(frame, 'Drain plug', (20, 50),
                                cv2.FONT_ITALIC, 0.6, [0,255,255], 1)
                    cls_ins.append(31)
        # Hand pose
        if 7 not in cycle_check and 15 in temp_list:
            f=0
            hand_i = [index for index, value in enumerate(temp_list) if value == 15]
            temp_ = []
            for ind in hand_i:
                hand_xyxy = temp_xyxy[ind]
                handx_c,handy_c = int(hand_xyxy[0]+abs(hand_xyxy[2]-hand_xyxy[0])/2),int(hand_xyxy[1]+abs(hand_xyxy[3]-hand_xyxy[1])/2)
                # cv2.circle()
                temp_.append([handx_c,handy_c])
                if 13 not in cycle_check and detent_box_pos[0]<handx_c<detent_box_pos[2] and detent_box_pos[1]<handy_c<detent_box_pos[3]:
                    # Hand in detent box
                    f = 1
                    if hand_f<3:
                        hand_f+=1
                    # if hand_f==3:
                    cycle_check.append(13)
                    print("detent box")
                    if spring_plungerf==1:
                        spring_plungerf = 0
                if 14 not in cycle_check and yellowS_box_pos[0]<handx_c<yellowS_box_pos[2] and yellowS_box_pos[1]<handy_c<yellowS_box_pos[3]:
                    # Hand in yellow spring box
                    f = 1
                    if hand_f<3:
                        hand_f+=1
                    # if hand_f==3:
                    cycle_check.append(14)
                    print("Yellow spring box")
                    if spring_plungerf==1:
                        spring_plungerf = 0
                if 15 not in cycle_check and pinkS_box_pos[0]<handx_c<pinkS_box_pos[2] and pinkS_box_pos[1]<handy_c<pinkS_box_pos[3]:
                    # Hand in pink spring box
                    f = 1
                    if hand_f<3:
                        hand_f+=1
                    # if hand_f==3:
                    cycle_check.append(15)
                    print("Pink spring box")
                    if spring_plungerf==1:
                        spring_plungerf = 0
            
            if f==0 and hand_f!=0:
                hand_f-=1
            if len(temp_)==2 and len(cycle_check)>0 and (cycle_check[-1] in [13,14,15]):
                temp_2 = math.dist(temp_[0],[354,273])
                temp_3 = math.dist(temp_[1],[354,273])
                print("1 hand dist",temp_2, temp_3)
                if 80<temp_2<100 and 80<temp_3<100:
                    temp_2 = math.dist(temp_[0],temp_[1])
                    print("2 hand dist",temp_2)
                    if 150<temp_2<210:
                        spring_plungerf = 1


    
    if len(cycle_check)>0 and cls_ins.count(13)>=3:
        cycle_check.clear()
        tower_conf.clear()
        sequence_break_f = 0

    frame = pos_results[0].plot(conf=False,labels=False,boxes=False)
    # frame = tower_results[0].plot()

    frame = cv2.resize(frame,(960,540))
    frame[510:510+logo.shape[0], 5:5+logo.shape[1]] = logo
    frame=pyg(frame,cycle_check)
    if rec_vid:
        video_rec.write(frame)

    return frame
    # return pyg(frame,cycle_check)
    

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
        # vid = cv2.VideoCapture("1724901275782.mp4") # 9
        # vid = cv2.VideoCapture("https://b2cb9bb9c7dd.ngrok.app/video_feed") # 9
    while True:
        try:
            frame = client.recv()
            # ret, frame = vid.read() 
            if frame is None:
                print("Client closed by None")
                client.close()
                break
            # print(time.time())
            # continue
            else:
                # frame = frame[211:700,394:915]
                frame_ori = frame.copy()
                # if not frame_queue.full():
                t = ThreadWithResult(
                    target=frameInf, args=(frame, time.time()))
                t.start()
                frame_queue.put(t)
                # if rec_vid:
                #     video_rec.write(frame)
                # cv2.waitKey(10)
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
    app.run(host='0.0.0.0', port=5000)


'''
#print(mcb_results[0].boxes.xyxy.tolist())
    
    temp_list2 = tyre_results[0].boxes.xyxy.tolist()

    if len(temp_list)>0: # and len(cls_ins)>0:
        for i in set(temp_list):
            cls_ins.append(i)
    elif len(cls_ins)>0:
        cls_ins.append(10)
    #if (temp_list.count(0)==5 or temp_list.count(2)==5) and cycle_check_f<5:
    # if (temp_list.count(3)>1) and cycle_check_f<5:
    #     print(temp_list)
    #     print(temp_list2)
    #     cycle_check_f+=1
    #     if cycle_check_f==5:
    #         temp_list2.sort()
    #         calibrate_posi(temp_list2)
    # elif cycle_check_f>=5:
    #     for i in screw_drill_pos:
    #         cv2.circle(frame,(i[0],i[1]),10,[255,0,0],2)

    # tyre present
    if cls_ins.count(3)>=conf_lev:
        # screw hole detected
        if cls_ins.count(0)>=conf_lev and temp_list.count(0)==5:
            if trigger_one_flag==0:
                temp = []
                for i,val in enumerate(temp_list):
                    if val==0:
                        temp.append(temp_list2[i])
                print(temp)
                temp.sort()
                calibrate_posi(temp)
                trigger_one_flag=1
        # for i in screw_drill_pos:
        #     cv2.circle(frame,(i[0],i[1]),10,[255,0,0],2)
        # screw detected
        if cls_ins.count(2)>=conf_lev and temp_list.count(2)==5 and len(cycle_check)==0:
            cycle_check.append(0)
            if trigger_one_flag==0:
                temp = []
                for i,val in enumerate(temp_list):
                    if val==2:
                        temp.append(temp_list2[i])
                print(temp)
                temp.sort()
                calibrate_posi(temp)
                trigger_one_flag=1
        # screw drill detected
        if cls_ins.count(1)>=conf_lev:
            try:
                x1,y1,x2,y2 = temp_list2[temp_list.index(1)]
                x_c,y_c = int(x1+abs(x2-x1)/2),int(y1+abs(y2-y1)/2)
                # cv2.circle(frame,(x_c,y_c),3,[0,0,255],-1)
                # print(x_c,y_c)

                for z,i in enumerate(screw_drill_pos):
                    dis = math.sqrt((x_c-i[0])**2+(y_c-i[1])**2)
                    # print("dis",dis)
                    if dis<15:
                        # cv2.circle(frame,(x_c,y_c),4,[0,255,0],-1)
                        # print("posi: ",z)
                        if (z+1) not in cycle_check:
                            print("************************************screw tightened***************************")
                            cycle_check.append(z+1)
                if cycle_original_sequence[0:len(cycle_check)] != cycle_check:
                    print("Sequence breaked")
                    if sequence_break_f==0:
                        sequence_break_f = len(cycle_check)
            except:
                pass

     
    elif cls_ins.count(3)==0:
        # tyre not present
        cycle_check.clear()
        screw_drill_conf.clear()
        sequence_break_f = 0
    # if temp_list.count(0)==5 and len(cycle_check)>0:
    #     cycle_check.clear()
    #     cycle_check_f=7
    # if (temp_list.count(2)==5) and len(cycle_check)==0:
    #     cycle_check.append(0)

    
    # elif len(cycle_check)==6:
    #     cycle_check_f=10
    #print(mcb_results[0].boxes.cls.tolist())
    # if len(cycle_check)>0:
    #     for i in screw_drill_pos:
    #         cv2.circle(frame,(i[0],i[1]),10,[255,0,0],2)
    frame = tyre_results[0].plot(conf=False,labels=False)
    frame = cv2.resize(frame,(960,540))
    frame[510:510+logo.shape[0], 5:5+logo.shape[1]] = logo
    # print("cycle: ",cycle_check)
    # print("cls ins: ",cls_ins)
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
'''