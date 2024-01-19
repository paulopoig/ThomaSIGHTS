from absl import flags
import sys
import os
from datetime import datetime
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from openpyxl import load_workbook

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')


max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Saving the Output Video or from Device's Capture Device
vid = cv2.VideoCapture('./data/video/students.mp4')

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/results.avi', codec, vid_fps, (vid_width, vid_height))


# Outside the loop
class_counts = {
    'clapping': 0,
    'tapping': 0,
    'bored': 0,
    'listening': 0,
    'curious': 0,
    'surprised': 0,
    'writing': 0,
    'leaning': 0,
    'sad': 0,
    'confused': 0,
    'focused': 0,
    'angry': 0,
    'scared': 0,
    'happy': 0,
}

# Historical Trajectory and Countings
from _collections import deque
pts = [deque(maxlen=30) for _ in range(1000)]


shades_of_purple_rgb = [
    (83, 91, 191),    # #535BBF Semi Deep Purple
    (122, 130, 233),  # #7A82E9 Lavander
    (71, 76, 138),    # #474C8A Deep Purple
    (74, 82, 191),    # #4A52BF Iconic Light Purple
    (140, 82, 255)    # #8C52FF Mmmmm Purple
]

# Convert RGB to BGR
shades_of_purple_bgr = [(color[2], color[1], color[0]) for color in shades_of_purple_rgb]

# Excel and CSV File for Database
excel_file_path = './data/database/test1.xlsx'
csv_file_path = './data/database/test1.csv'

# Set to keep track of unique detection IDs
unique_detection_ids = set()

# Outside the loop
if os.path.exists(excel_file_path):
    historical_data = pd.read_excel(excel_file_path)
else:
    # If the file doesn't exist, create an empty DataFrame
    historical_data = pd.DataFrame(columns=[' date ', '  time '] + list(class_counts.keys()))

while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break


    # Insert a semi-transparent rectangle at the top
    rectangle_color = (240, 240, 240)  # Black color
    rectangle_height = 130
    alpha = 0  # Adjust the alpha value for transparency

    # Create a transparent overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (vid_width, rectangle_height), rectangle_color, cv2.FILLED)

    # Blend the overlay with the original image
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

# Starting the Timer
    t1 = time.time()

    boxes, scores, classes, nums = yolo.predict(img_in)

# Boxes, 3D shape (1, 100, 4) X, Y coordinates and Width, Height
# Scores, 2D shape (1, 100)
# Classes, 2D shape (1, 100)
# Nums, 1D shape (1,)

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections]) # Class Objects
    scores = np.array([d.confidence for d in detections]) # Confidence Scores
    classes = np.array([d.class_name for d in detections])

    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections) # Will Update Kalman Tracker Parameters and Features Set

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

# Make Current Count Counter
    #current_count = int(0)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        bbox = track.to_tlbr()
        class_name = track.get_class()
        # Assign shades of purple based on track ID
        shade_index = int(track.track_id) % len(shades_of_purple_bgr)
        color = shades_of_purple_bgr[shade_index]

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                    +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2) # We are going to put the text somewhere in the middle of this rectangles with the white color

# In this line includes how to incorporate Hystorical Trajectory in MOT
        #Center Point
        center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
        pts[track.track_id].append(center)

        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

        #height, width, _ = img.shape
        #cv2.line(img, (0, int(3 * height / 6 + height/ 20)), (width, int(3 * height / 6 + height / 20)), (0, 255, 0), thickness=2)
        #cv2.line(img, (0, int(3 * height / 6 - height/ 20)), (width, int(3 * height / 6 - height / 20)), (0, 255, 0), thickness=2)

        center_y = int(((bbox[1]) + (bbox[3])) / 2)

        class_name_lower = class_name.lower()
        if class_name_lower in class_counts and track.track_id not in unique_detection_ids:
            class_counts[class_name_lower] += 1
            unique_detection_ids.add(track.track_id)


        print("Class Counts:", class_counts)
        print("Detection ID:", track.track_id)
        print("Number of Detections:", nums)


    fps = 1./(time.time()-t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30), 0, 1, (71,76,138), 2)
    cv2.namedWindow('output')
    cv2.resizeWindow('output', 1024, 768)
    cv2.imshow('output', img)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break


# Save the final counts to the Excel file outside the loop
if nums > 0:  # Only update if there are detections
    current_datetime = datetime.now()
    data = {
        ' date ': [current_datetime.strftime("%Y-%m-%d")],
        ' time ': [current_datetime.strftime("%H:%M:%S")],
    }
    data.update(class_counts)

    historical_data = historical_data.append(data, ignore_index=True)

    # Reset index before saving to Excel
    historical_data.reset_index(drop=True, inplace=True)

    # Save to a new Excel file with startcol=1
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        # Save the DataFrame to Excel with startcol=1 and index=False
        historical_data.to_excel(writer, index=False, sheet_name='Sheet1')

    # Save to CSV
    historical_data.to_csv(csv_file_path, index=False)
