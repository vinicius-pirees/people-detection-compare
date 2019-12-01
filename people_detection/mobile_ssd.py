import cv2
import os, sys
import numpy as np
import time
import subprocess
from matplotlib import pyplot as plt
from utils import tile_image, append_tiles_image, tiles_info, box_new_coords, detect_over_frames, non_max_suppression, write_cpu_usage_file
from tqdm import tqdm
import json
import psutil
import argparse

parser = argparse.ArgumentParser(description='Detection - Mobile SSD')

parser.add_argument('-d', '--detect-on-tiles',
                    dest='detect_on_tiles',
                    type=str,
                    default='n')
parser.add_argument('-p', '--debug',
                    dest='debug',
                    type=str,
                    default='n')
parser.add_argument('-x', '--tiles-x',
                    dest='tiles_x',
                    type=int)
parser.add_argument('-y', '--tiles-y',
                    dest='tiles_y',
                    type=int)
parser.add_argument('-m', '--margin-percent',
                    dest='margin_percent',
                    type=float,
                    default=0.25)
parser.add_argument('-o', '--output-video',
                    dest='output_video',
                    type=str,
                    default='n')
parser.add_argument('-s', '--start-frame',
                    dest='start_frame',
                    type=int)
parser.add_argument('-e', '--end-frame',
                    dest='end_frame',
                    type=int)
parser.add_argument('-f', '--only-moving-frames',
                    dest='only_moving_frames',
                    type=str)
parser.add_argument('-c', '--confidence',
                    dest='confidence',
                    type=float,
                    default=0.2)

args = parser.parse_args()
detect_on_tiles = args.detect_on_tiles
debug = args.debug
tiles_x = args.tiles_x
tiles_y = args.tiles_y
margin_percent = args.margin_percent
output_video = args.output_video
start_frame = args.start_frame
end_frame = args.end_frame
only_moving_frames = args.only_moving_frames
confidence = args.confidence

main_dir = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.join(main_dir, 'motion_detection'))

# detect_on_tiles = 'y'
# debug = 'n'
# only_moving_frames = 'y'
# tiles_x = 6
# tiles_y = 7
# output_video = 'n'
# confidence = 0.2
# margin_percent = 0.25
# # start_frame = 550
# # end_frame = 800
# start_frame = None
# end_frame = None

videos_path = '/home/vgoncalves/personal-git/people-detection-compare/resources/virat_dataset/'

video_name_list = ['VIRAT_S_000201_02_000590_000623',
                   'VIRAT_S_010000_00_000000_000165',
                   'VIRAT_S_010003_01_000111_000137',
                   'VIRAT_S_010106_01_000493_000526',
                   'VIRAT_S_010200_03_000470_000567',
                   'VIRAT_S_050000_12_001591_001619']


mobile_ssd_dir = os.path.join(main_dir, 'models/mobile_ssd')
prototxt = os.path.join(mobile_ssd_dir, 'MobileNetSSD_deploy.prototxt.txt')
model = os.path.join(mobile_ssd_dir, 'MobileNetSSD_deploy.caffemodel')

classes_path = os.path.sep.join([mobile_ssd_dir, "classes"])
MOBILE_SSD_CLASSES = open(classes_path).read().strip().split("\n")

mobile_ssd_net = cv2.dnn.readNetFromCaffe(prototxt, model)


if only_moving_frames == 'y':
    ground_truth_file = os.path.join(main_dir, 'resources/virat_dataset/VIRAT_videos_moving_ground_truth_boxes.json')
else:
    ground_truth_file = os.path.join(main_dir, 'resources/virat_dataset/VIRAT_videos_ground_truth_boxes.json')

with open(ground_truth_file) as infile:
    ground_truth_boxes = json.load(infile)

moving_frames = None
if only_moving_frames == 'y':
    moving_frames_file = main_dir + '/results/VIRAT_videos_moving_frames.json'
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


video_dict = {}
total_frames_all_videos = 0

for video_name in video_name_list:
    video_file_path = videos_path + video_name + '.mp4'

    tiles_dict = None
    if detect_on_tiles == 'y':
        vs = cv2.VideoCapture(video_file_path)  # Get tile size
        success, frame = vs.read()
        tiles_dict = tiles_info(frame, tiles_x, tiles_y, margin_percent)

    vs = cv2.VideoCapture(video_file_path)  # Video
    num_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    # if start_frame is None:
    #     start_frame = 0
    # if end_frame is None:
    #     end_frame = num_frames
    start_frame = 0
    end_frame = num_frames

    #vs.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Read from start frame
    total_frames = end_frame - start_frame
    current_frame = start_frame
    total_frames_all_videos += total_frames
    video_dict[video_name] = {}
    video_dict[video_name]['total_frames'] = total_frames
    video_dict[video_name]['tiles_dict'] = tiles_dict
    video_dict[video_name]['start_frame'] = start_frame
    video_dict[video_name]['end_frame'] = end_frame
    video_dict[video_name]['video_file_path'] = video_file_path

current_process = psutil.Process()
print(current_process)

command = 'python ' + main_dir + '/people_detection/get_cpu_usage.py --pid ' + str(current_process.pid)

print(command)
#process = subprocess.Popen([command], stdout=None, stderr=None,shell=True, preexec_fn=os.setpgrp)
os.system('nohup ' + command + ' &')


detect_over_frames(video_dict,
                       'mobile_ssd',
                       detect_single_frame,
                       output_video=output_video,
                       detect_on_tiles=detect_on_tiles,
                       debug=debug,
                       tiles_x=tiles_x,
                       tiles_y=tiles_y,
                       model=mobile_ssd_net,
                       ground_truth_boxes=ground_truth_boxes,
                       main_dir=main_dir,
                       confidence=confidence,
                       only_moving_frames=only_moving_frames,
                       total_frames_all_videos=total_frames_all_videos,
                       moving_frames=moving_frames)


if detect_on_tiles == 'y':
    write_cpu_usage_file(main_dir, 'VIRAT_videos', 'mobile_ssd', str(tiles_x) + ',' + str(tiles_y))
else:
    write_cpu_usage_file(main_dir, 'VIRAT_videos', 'mobile_ssd', None)