import cv2
import numpy as np
import csv
import time
import concurrent.futures
from datetime import datetime
import glob
import os
from button_check import capture_image_ds

# Function to read CSV values from a specified path
def read_csv_values_from_path(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        values = next(reader)
    return values

# Function to read paths from master CSV file
def read_master_csv(master_csv_path):
    paths = {}
    with open(master_csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            key, path = row
            paths[key] = path
    return paths

# Function to automatically adjust brightness and contrast
def automatic_brightness_and_contrast(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    accumulator = np.cumsum(hist)

    maximum = accumulator[-1]
    clip_hist_percent = 0.01 * maximum
    minimum_gray = np.argmax(accumulator > clip_hist_percent)
    maximum_gray = 255 - np.argmax(accumulator[::-1] > clip_hist_percent)
    if maximum_gray == minimum_gray:
        print("Error: Cannot adjust brightness and contrast due to low contrast in the image.")
        return image
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return auto_result

# Function to capture an image from the camera
def capture_image(camera_index=0, warmup_delay=1):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return None

    ret, frame = cap.read()
    if not ret or not frame.any():
        print("Initial capture failed, warming up the camera...")
        time.sleep(warmup_delay)
        ret, frame = cap.read()

    cap.release()

    if ret and frame.any():
        return frame
    else:
        return None

# Function to process template matching
def process_template_matching(template, threshold, auto_result):
    res = cv2.matchTemplate(auto_result, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    match_count = len(loc[0])

    for pt in zip(*loc[::-1]):
        cv2.rectangle(auto_result, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 255), 1)
        #cv2.putText(auto_result, "OK" if match_count > 0 else "NG", (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0) if match_count > 0 else (0, 0, 255), 1, cv2.LINE_AA)

    return match_count

# Function to get the current serial number from a file
def get_current_serial_number(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            return len(lines) + 1
    return 1

# Read paths from master CSV
master_csv_path = "master_file.csv"
paths = read_master_csv(master_csv_path)

# Read values from CSV files specified in the master CSV
threshold_file_path = paths["FHW_Thresh_Value"]
crop_file_path = paths["FHW_Crop_Value"]
template_folder = paths["templates_folder"]

threshold_values = read_csv_values_from_path(threshold_file_path)
crop_values = read_csv_values_from_path(crop_file_path)

threshold_FR1 = float(threshold_values[0])
crop_FR1_x1, crop_FR1_x2, crop_FR1_y1, crop_FR1_y2 = map(int, crop_values[1:5])

# Capture an image from the camera
frame = capture_image_ds()

if frame is None:
    print("Error: Captured frame is black.")
    exit()

img_rgb = frame

# Perform automatic brightness and contrast adjustment
img_crop = img_rgb[crop_FR1_y1:crop_FR1_y2, crop_FR1_x1:crop_FR1_x2]
auto_result = automatic_brightness_and_contrast(img_crop)

# Load templates from the folder specified in the master CSV
template_paths = glob.glob(os.path.join(template_folder, '*.jpg')) + glob.glob(os.path.join(template_folder, '*.png'))
templates = [cv2.imread(template_path) for template_path in template_paths]

threshold = 0.85
match_counts = []

# Perform template matching in parallel
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = [executor.submit(process_template_matching, template, threshold, auto_result.copy()) for template in templates]
    for result in results:
        match_counts.append(result.result())

total_rectangles = sum(match_counts)
print(f"Total number of rectangles: {total_rectangles}")

Result = "OK" if total_rectangles == 2 else "NG"
cv2.imwrite('Processed.jpg', auto_result)

# Get the current timestamp
timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

# Get the current serial numbers for the files
serial_number_count = get_current_serial_number('Count.txt')
serial_number_result = get_current_serial_number('Result.txt')

# Write results to text files
with open('Count.txt', 'a') as f:
    f.write(f'{serial_number_count}. [{timestamp}] Number of Gaskets present: {total_rectangles}\n')

with open('Result.txt', 'a') as g:
    g.write(f'{serial_number_result}. [{timestamp}] This is a - {Result} part\n')

# Resize the processed image to fit a 7-inch screen
screen_width = 7 * 25.4  # 7 inches in millimeters
screen_height = 7 * 25.4 * (10 / 16)  # Assuming 16:10 aspect ratio
resized_image = cv2.resize(auto_result, (int(screen_width), int(screen_height)))    

# Display the processed image for 10 seconds
#cv2.imshow('Sample', auto_result)
cv2.imshow('Processed Image', resized_image)
cv2.waitKey(10000)  # 10 seconds
cv2.destroyAllWindows()

