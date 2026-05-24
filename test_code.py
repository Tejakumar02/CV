# # importing libraries
# import cv2
# import time 

# # Create a VideoCapture object and read from input file
# cap = cv2.VideoCapture('rtsp://localhost:8554/stream')

# # Check if camera opened successfully
# if (cap.isOpened()== False):
#     print("Error opening video file")

# # Read until video is completed
# while(cap.isOpened()):
    
# # Capture frame-by-frame
#     ret, frame = cap.read()
#     if ret == True:
#     # Display the resulting frame
#         cv2.imshow('Frame', frame)
#         print(time.time())
#     # Press Q on keyboard to exit
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break

# # Break the loop
#     else:
#         break

# # When everything done, release
# # the video capture object
# cap.release()

# # Closes all the frames
# cv2.destroyAllWindows()



# .pt to onnx convertion 
# import sys
# sys.path.append("C:/AL_chennai_tower/yolov10")

# from ultralytics import YOLOv10
# from ultralytics import YOLO
# # Load a model
# model = YOLO('models/AL_tower_081124.pt',task='pose')  # load an official model

# # Export the model
# model.export(format='onnx')


# import os
# import cv2
# import glob
# import sys
# sys.path.append("C:/all_codes/yolov10")

# from ultralytics import YOLOv10
# from ultralytics import YOLO
# model = YOLOv10('C:/Users/Administrator/Downloads/MS_delhi_bonnet3_220924.pt', task="detect")
# for name in glob.glob('C:/Users/Administrator/Downloads/test/*.png'):
#     frame = cv2.imread(name)
#     pos_results = model.predict(
#             frame, conf=0.1, iou=0.2, imgsz=640, verbose=False,device =0)
#     frame = pos_results[0].plot(conf=True, labels = True)
#     cv2.imshow("result",frame)
#     cv2.waitKey(0)

# import onnxruntime as ort
# # Print the available providers
# print(ort.get_available_providers())

# import onnxruntime as ort
# providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
# session = ort.InferenceSession("models/AL_tower_081124.onnx", providers=providers)
import ultralytics
ultralytics.checks()