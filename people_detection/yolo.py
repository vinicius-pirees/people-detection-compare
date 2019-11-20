import cv2
import os
import numpy as np
import time
from matplotlib import pyplot as plt
from utils import tile_image, append_tiles_image, tiles_info, box_new_coords, detect_over_frames
from tqdm import tqdm
import json


detect_on_tiles = 'n'
debug = 'y'
tiles_x = 2
tiles_y = 3
confidence = 0.5
threshold = 0.3
video_file = '/home/vgoncalves/personal-git/people_detection_compare/resources/virat_dataset/VIRAT_S_010000_00_000000_000165.mp4'
output_video = 'n'
margin_percent = 0.25
start_frame = 550
end_frame = 570
# start_frame = None
# end_frame = None

video_file_name = os.path.splitext(os.path.split(video_file)[1])[0]
main_dir = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


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


ground_truth_file = main_dir + '/resources/virat_dataset/' + video_file_name + '_ground_truth_boxes.json'

with open(ground_truth_file) as infile:
    ground_truth_boxes = json.load(infile)



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
                   'yolo',
                   detect_single_frame,
                   output_video=output_video,
                   detect_on_tiles=detect_on_tiles,
                   debug=debug,
                   tiles_x=tiles_x,
                   tiles_y=tiles_y,
                   model=yolo_net,
                   ground_truth_boxes=ground_truth_boxes,
                   current_frame=current_frame,
                   end_frame=end_frame,
                   total_frames=total_frames,
                   main_dir=main_dir,
                   video_file_name=video_file_name,
                   confidence=confidence,
                   threshold=threshold,
                   tiles_dict=tiles_dict)