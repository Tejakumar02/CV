from datetime import datetime
import requests
import time
import threading
loc = threading.Lock()
import cv2
uframe=cv2.imread("/home/pi/all_codes/defect_scanner_logo.png")
def collect_frame():
    vid = cv2.VideoCapture("rtsp://sourab:tvsm123!@192.168.1.101/stream2")
    count=0
    while(True):
        ret, frame = vid.read()
        if ret == False:
            break
        frame=frame[330:410,546:700]
        frame = cv2.resize(frame,(640,360))
        if count==0:
            cv2.imwrite("/home/pi/all_codes/d4.png",frame)
            count+=1
        cv2.imshow("Live Cam",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()


if __name__ =="__main__":
    #t1 = threading.Thread(target=send_frame)
    t2 = threading.Thread(target=collect_frame)
 
    #t1.start()
    t2.start()
