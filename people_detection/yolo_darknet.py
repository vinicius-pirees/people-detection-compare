import cv2
import os
import sys

main_dir = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
yolo_dir = os.path.join(main_dir, 'models/yolo')
darknet_dir = os.path.join(main_dir, 'models/darknet')
darknet_script_dir = os.path.join(main_dir, 'models/darknet_python')

sys.path.append(darknet_script_dir)
import darknet
import numpy as np
import time
from matplotlib import pyplot as plt
from utils import tile_image, append_tiles_image, tiles_info, box_new_coords, detect_over_frames, non_max_suppression, write_cpu_usage_file
from tqdm import tqdm
import json
import psutil
import argparse

darknet.set_gpu(3)
parser = argparse.ArgumentParser(description='Detection - YOLO')

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
                    default=0.5)
parser.add_argument('-t', '--threshold',
                    dest='threshold',
                    type=float,
                    default=0.45)

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
threshold = args.threshold



# detect_on_tiles = 'y'
# debug = 'y'
# only_moving_frames = 'y'
# tiles_x = 4
# tiles_y = 5
# confidence = 0.5
# threshold = 0.3
# output_video = 'n'
# margin_percent = 0.25
# start_frame = 550
# end_frame = 570
# # start_frame = None
# # end_frame = None




videos_path = os.path.join(main_dir,'resources/virat_dataset/')

with open(os.path.join(videos_path, 'videos_to_process.txt')) as f:
    video_list = f.read().splitlines()



video_name_list = [video.split('.')[0] for video in video_list]



cfg_path = os.path.join(yolo_dir, "yolov3.cfg")
weights_path = os.path.join(yolo_dir, "yolov3.weights")
meta_path = os.path.join(yolo_dir,"coco.data")

print(cfg_path)
print(weights_path)

net = darknet.load_net(bytes(cfg_path, encoding='utf-8'), bytes(weights_path, encoding='utf-8'), 0)
meta = darknet.load_meta(bytes(meta_path, encoding='utf-8'))



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



def detect_single_frame(yolo_net, frame, row, column, tiles_dict, **kwargs):
    confidence = kwargs.get('confidence')
    threshold = kwargs.get('threshold')

    start_time = time.time()
    detections = darknet.detect(yolo_net, meta, frame, nms=threshold)
    total_time = time.time() - start_time

    boxes = []
    confidences = []
    frame_with_boxes = frame.copy()

    for detection in detections:
        _class = detection[0].decode("utf-8")
        if _class == 'person':
            _confidence = detection[1]
            if _confidence > confidence:
                confidences.append(_confidence)

                (centerX, centerY, width, height) = detection[2]
                # (startX, startY, endX, endY) = detection[2]

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                box = [int(x), int(y), int(width), int(height)]

                boxes.append(box)

                if tiles_dict is not None:
                    startX, startY, endX, endY = box_new_coords(box,
                                                                row,
                                                                column,
                                                                tiles_dict)
                else:
                    (startX, startY, endX, endY) = box

                cv2.rectangle(frame_with_boxes, (startX, startY), (endX, endY), (0, 255, 0), 2)

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
    #     end_frame = num_framescl
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
os.system('nohup ' + command + ' &')

detect_over_frames(video_dict,
                   'yolo_darknet',
                   detect_single_frame,
                   output_video=output_video,
                   detect_on_tiles=detect_on_tiles,
                   debug=debug,
                   tiles_x=tiles_x,
                   tiles_y=tiles_y,
                   model=net,
                   ground_truth_boxes=ground_truth_boxes,
                   main_dir=main_dir,
                   only_moving_frames=only_moving_frames,
                   total_frames_all_videos=total_frames_all_videos,
                   moving_frames=moving_frames,
                   confidence=confidence,
                   threshold=threshold)

if detect_on_tiles == 'y':
    write_cpu_usage_file(main_dir, 'VIRAT_videos', 'yolo_darknet', str(tiles_x) + ',' + str(tiles_y))
else:
    write_cpu_usage_file(main_dir, 'VIRAT_videos', 'yolo_darknet', None)
