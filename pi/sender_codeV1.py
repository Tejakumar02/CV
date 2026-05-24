import cv2
from vidgear.gears import VideoGear
from vidgear.gears import NetGear
from datetime import datetime
import requests
import time
import threading
loc = threading.Lock()
uframe = cv2.imread("/home/pi/all_codes/defect_scanner_logo.png")


def update_frame(frame):
    global uframe
    global loc
    loc.acquire()
    uframe = frame
    loc.release()


def collect_frame():
    vid = cv2.VideoCapture("rtsp://sourab:tvsm123!@192.168.1.101/stream2")
    while (True):
        ret, frame = vid.read()
        try:
            frame.shape
            update_frame(frame)
        except Exception as e:
            print("No frame to upload", e)


def send_frame():
    global uframe
    global loc
    try:
        # requests.get("http://98.70.57.180:5000/img_update",timeout=2)
        # https://deploy.defectscanner.com:5000/video_feed http://al.defectscanner.com:5000/video_feed
        requests.get(
            "http://al.defectscanner.com:5000/img_update", timeout=2)
        # pass
    except:
        pass
    options = {"max_retries": 130, "CAP_PROP_FRAME_WIDTH": 960, "CAP_PROP_FRAME_HEIGHT": 540, "CAP_PROP_FPS": 30,
               "jpeg_compression": True,
               "jpeg_compression_quality": 90,
               "jpeg_compression_fastdct": True,
               "jpeg_compression_fastupsample": True, }
    server = NetGear(address="98.70.80.22", port="12345", protocol="tcp",
                     pattern=1, **options,)  # Define netgear server with default settings
    # stream = VideoGear(source="rtsp://sourab:tvsm123!@192.168.3.11/stream2").start()
    # stream=cv2.VideoCapture("rtsp://sourab:tvsm123!@192.168.3.11/stream2")
    # infinite loop until [Ctrl+C] is pressed
    while True:
        try:
            # frame = picam2.capture_array()#stream.read()
            # read frames
            loc.acquire()
            frame = uframe.copy()
            loc.release()
            # _,frame = stream.read()
            # check if frame is None
            if frame is None:
                # if True break the infinite loop
                continue
            frame = cv2.resize(frame, (960, 540))
            # cv2.imshow("re",frame)
            # cv2.waitKey(1)
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            cv2.putText(frame, f"Time:{current_time}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, [
                        255, 255, 255], 2, cv2.LINE_AA)

            # send frame to server
            server.send(frame)

        except KeyboardInterrupt:
            # break the infinite loop
            break
    print("successfully closed")
    # safely close video stream
    # stream.stop()
    # safely close server
    server.close()


if __name__ == "__main__":
    t1 = threading.Thread(target=send_frame)
    t2 = threading.Thread(target=collect_frame)

    t1.start()
    t2.start()
