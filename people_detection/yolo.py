import cv2
import os
import numpy as np
import time
from matplotlib import pyplot as plt
from utils import tile_image, append_tiles_image, tiles_info, box_new_coords, detect_over_frames, non_max_suppression, write_cpu_usage_file
from tqdm import tqdm
import json
import psutil
import argparse

parser = argparse.ArgumentParser(description='Detection - HOG')

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
                    default=0.3)

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


main_dir = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

videos_path = os.path.join(main_dir,'resources/virat_dataset/')

with open(os.path.join(videos_path, 'videos_to_process.txt')) as f:
    video_list = f.read().splitlines()

video_name_list = [video.split('.')[0] for video in video_list]


yolo_dir = os.path.join(main_dir, 'models/yolo')
labelsPath = os.path.sep.join([yolo_dir, "coco.names"])
YOLO_LABELS = open(labelsPath).read().strip().split("\n")
# # initialize a list of colors to represent each possible class label
# np.random.seed(42)
# YOLO_COLORS = np.random.randint(0, 255, size=(len(YOLO_LABELS), 3), dtype="uint8")
weightsPath = os.path.sep.join([yolo_dir, "yolov3-tiny.weights"])
configPath = os.path.sep.join([yolo_dir, "yolov3-tiny.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
yolo_net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = yolo_net.getLayerNames()
ln = [ln[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]  # determine only the *output* layer names that we need from YOLO



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
    (H, W) = frame.shape[:2]

    confidence = kwargs.get('confidence')
    threshold = kwargs.get('threshold')

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    start_time = time.time()
    layerOutputs = yolo_net.forward(ln)
    total_time = time.time() - start_time

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    final_boxes = []
    final_confidences = []
    frame_with_boxes = frame.copy()

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            _confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if _confidence > confidence:
                if YOLO_LABELS[int(classID)] == 'person':
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(_confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                box = [x, y, (x + w), (y + h)]
                confidence_c = confidences[i]

                if tiles_dict is not None:
                    startX, startY, endX, endY = box_new_coords(box,
                                                                row,
                                                                column,
                                                                tiles_dict)
                else:
                    (startX, startY, endX, endY) = box

                final_boxes.append([int(startX), int(startY), int(endX), int(endY)])
                final_confidences.append(float(confidence_c))


                # draw a bounding box rectangle and label on the frame
                cv2.rectangle(frame_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "{}: {:.4f}".format(YOLO_LABELS[classIDs[i]],
                                           confidences[i])
                cv2.putText(frame_with_boxes, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return final_boxes, final_confidences, total_time, frame_with_boxes


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
os.system('nohup ' + command + ' &')


detect_over_frames(video_dict,
                   'yolo',
                   detect_single_frame,
                   output_video=output_video,
                   detect_on_tiles=detect_on_tiles,
                   debug=debug,
                   tiles_x=tiles_x,
                   tiles_y=tiles_y,
                   model=yolo_net,
                   ground_truth_boxes=ground_truth_boxes,
                   main_dir=main_dir,
                   only_moving_frames=only_moving_frames,
                   total_frames_all_videos=total_frames_all_videos,
                   moving_frames=moving_frames,
                   confidence=confidence,
                   threshold=threshold)

if detect_on_tiles == 'y':
    write_cpu_usage_file(main_dir, 'VIRAT_videos', 'yolo', str(tiles_x) + ',' + str(tiles_y))
else:
    write_cpu_usage_file(main_dir, 'VIRAT_videos', 'yolo', None)
