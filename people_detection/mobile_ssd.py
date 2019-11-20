import cv2
import os, sys
import numpy as np
import time
from matplotlib import pyplot as plt
from utils import tile_image, append_tiles_image, tiles_info, box_new_coords, detect_over_frames
from tqdm import tqdm
import json

main_dir = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(main_dir, 'motion_detection'))

detect_on_tiles = 'y'
debug = 'n'
only_moving_frames = 'y'
tiles_x = 3
tiles_y = 4
video_file = '/home/vgoncalves/personal-git/people_detection_compare/resources/virat_dataset/VIRAT_S_010000_00_000000_000165.mp4'
output_video = 'n'
confidence = 0.2
margin_percent = 0.25
# start_frame = 550
# end_frame = 570
start_frame = None
end_frame = None

video_file_name = os.path.splitext(os.path.split(video_file)[1])[0]

mobile_ssd_dir = os.path.join(main_dir, 'models/mobile_ssd')
prototxt = os.path.join(mobile_ssd_dir, 'MobileNetSSD_deploy.prototxt.txt')
model = os.path.join(mobile_ssd_dir, 'MobileNetSSD_deploy.caffemodel')

classes_path = os.path.sep.join([mobile_ssd_dir, "classes"])
MOBILE_SSD_CLASSES = open(classes_path).read().strip().split("\n")

mobile_ssd_net = cv2.dnn.readNetFromCaffe(prototxt, model)

ground_truth_file = main_dir + '/resources/virat_dataset/' + video_file_name + '_ground_truth_boxes.json'

with open(ground_truth_file) as infile:
    ground_truth_boxes = json.load(infile)

if only_moving_frames == 'y':
    moving_frames_file = main_dir + '/results/' + video_file_name + '_moving_frames.json'
    with open(moving_frames_file) as infile:
        moving_frames = json.load(infile)


def detect_single_frame(model, frame, row, column, tiles_dict, **kwargs):
    (h, w) = frame.shape[:2]
    confidence = kwargs.get('confidence')
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    model.setInput(blob)

    start_time = time.time()
    detections = model.forward()
    total_time = time.time() - start_time

    boxes = []
    confidences = []
    frame_with_boxes = frame.copy()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        _confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if _confidence > confidence:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            if MOBILE_SSD_CLASSES[idx] == 'person':

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

                if tiles_dict is not None:
                    startX, startY, endX, endY = box_new_coords(box.astype("int"),
                                                                row,
                                                                column,
                                                                tiles_dict)
                else:
                    (startX, startY, endX, endY) = box.astype("int")

                boxes.append([int(startX), int(startY), int(endX), int(endY)])
                confidences.append(float(_confidence))

                label = "{:.2f}%".format(_confidence * 100)
                cv2.rectangle(frame_with_boxes, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame_with_boxes, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
                   'mobile_ssd',
                   detect_single_frame,
                   output_video=output_video,
                   detect_on_tiles=detect_on_tiles,
                   debug=debug,
                   tiles_x=tiles_x,
                   tiles_y=tiles_y,
                   model=mobile_ssd_net,
                   ground_truth_boxes=ground_truth_boxes,
                   current_frame=current_frame,
                   end_frame=end_frame,
                   total_frames=total_frames,
                   main_dir=main_dir,
                   video_file_name=video_file_name,
                   confidence=confidence,
                   tiles_dict=tiles_dict,
                   moving_frames=moving_frames)