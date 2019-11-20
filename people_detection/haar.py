import cv2
import os
import numpy as np
import time
from matplotlib import pyplot as plt
from utils import tile_image, append_tiles_image, tiles_info, box_new_coords, detect_over_frames
from tqdm import tqdm
import json


detect_on_tiles = 'n'
debug = 'n'
tiles_x = 3
tiles_y = 4
scaleFactor = 1.1
minNeighbors = 3
video_file = '/home/vgoncalves/personal-git/people_detection_compare/resources/virat_dataset/VIRAT_S_010000_00_000000_000165.mp4'
output_video = 'n'
margin_percent = 0.25
start_frame = 550
end_frame = 570
# start_frame = None
# end_frame = None

video_file_name = os.path.splitext(os.path.split(video_file)[1])[0]
main_dir = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

full_body_cascade = cv2.CascadeClassifier(os.path.join(main_dir, 'models/haar/haarcascade_fullbody.xml'))

ground_truth_file = main_dir + '/resources/virat_dataset/' + video_file_name + '_ground_truth_boxes.json'

with open(ground_truth_file) as infile:
    ground_truth_boxes = json.load(infile)


def detect_single_frame(full_body_cascade, frame, row, column, tiles_dict, **kwargs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    scaleFactor = kwargs.get('scaleFactor')
    minNeighbors = kwargs.get('minNeighbors')

    start_time = time.time()
    bodies = full_body_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)
    total_time = time.time() - start_time

    boxes = []
    confidences = []
    frame_with_boxes = frame.copy()

    # Draw rectangle around the faces
    for (x, y, w, h) in bodies:
        if tiles_dict is not None:
            startX, startY, endX, endY = box_new_coords([x, y, (x + w), (y + h)],
                                                        row,
                                                        column,
                                                        tiles_dict)
        else:
            (startX, startY, endX, endY) = (x, y, (x + w), (y + h))

        cv2.rectangle(frame_with_boxes, (startX, startY), (endX, endY), (0, 255, 0), 2)

        boxes.append([int(startX), int(startY), int(endX), int(endY)])
        confidences.append(1)

    return boxes, confidences, total_time, frame_with_boxes


tiles_dict = None
if detect_on_tiles == 'y':
    vs = cv2.VideoCapture(video_file)  # Get tile size
    success, frame = vs.read()
    tiles_dict = tiles_info(frame, tiles_x, tiles_y, margin_percent)

vs = cv2.VideoCapture(video_file)  # Video
num_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
if start_frame is None:
    start_frame = 0
if end_frame is None:
    end_frame = num_frames

vs.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Read from start frame
total_frames = end_frame - start_frame
current_frame = start_frame
writer = None
dict_predictions = {}
progress_bar = tqdm(total=total_frames)
frames_times = []

detect_over_frames(vs,
                   'haar',
                   detect_single_frame,
                   output_video=output_video,
                   detect_on_tiles=detect_on_tiles,
                   debug=debug,
                   tiles_x=tiles_x,
                   tiles_y=tiles_y,
                   model=full_body_cascade,
                   ground_truth_boxes=ground_truth_boxes,
                   current_frame=current_frame,
                   end_frame=end_frame,
                   total_frames=total_frames,
                   main_dir=main_dir,
                   video_file_name=video_file_name,
                   scaleFactor=scaleFactor,
                   minNeighbors=minNeighbors,
                   tiles_dict=tiles_dict)