import cv2
import os
import numpy as np
import time
from matplotlib import pyplot as plt
from utils import tile_image, append_tiles_image, tiles_info, box_new_coords
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


def detect_single_frame(full_body_cascade, frame, scaleFactor=1.1, minNeighbors=3, tile_info=None, tiles_dict=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    start_time = time.time()
    bodies = full_body_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)
    total_time = time.time() - start_time

    boxes = []
    confidences = []
    frame_with_boxes = frame.copy()

    # Draw rectangle around the faces
    for (x, y, w, h) in bodies:
        if tile_info is not None:
            startX, startY, endX, endY = box_new_coords([x, y, (x + w), (y + h)],
                                                        tile_info['row'],
                                                        tile_info['column'],
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


while True:
    success, frame = vs.read()
    all_boxes = []
    all_confidences = []
    all_times = []

    if not success:
        vs.release()
        cv2.destroyAllWindows()
        if output_video == 'y':
            writer.release()
        break

    if detect_on_tiles == 'y':
        tiles = tile_image(frame, tiles_x, tiles_y, tiles_dict)

        if debug == 'y':
            # Debug a specific tile
            row = 1
            column = 1
            boxes, confidences, total_time, final_frame = detect_single_frame(full_body_cascade,
                                                                              tiles[row, column],
                                                                              scaleFactor=scaleFactor,
                                                                              minNeighbors=minNeighbors
                                                                              )

        else:
            final_frame = frame.copy()

            for row in range(0, tiles_x):
                for column in range(0, tiles_y):
                    boxes, confidences, total_time, _ = detect_single_frame(full_body_cascade,
                                                                              tiles[row, column],
                                                                              scaleFactor=scaleFactor,
                                                                              minNeighbors=minNeighbors,
                                                                              tile_info={
                                                                                  'row': row,
                                                                                  'column': column},
                                                                              tiles_dict=tiles_dict)

                    all_boxes = all_boxes + boxes
                    all_confidences = all_confidences + confidences
                    all_times.append(total_time)

            for box in all_boxes:
                (startX, startY, endX, endY) = box

                cv2.rectangle(final_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            for box in ground_truth_boxes['frame_' + str(current_frame)]:
                (startX, startY, endX, endY) = box

                cv2.rectangle(final_frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

            frames_times.append(np.sum(all_times))

    else:  # See entire frame
        if debug == 'y':
            boxes, confidences, total_time, final_frame = detect_single_frame(full_body_cascade,
                                                                              frame,
                                                                              scaleFactor=scaleFactor,
                                                                              minNeighbors=minNeighbors
                                                                              )
        else:
            boxes, confidences, total_time, _ = detect_single_frame(full_body_cascade,
                                                                    frame,
                                                                    scaleFactor=scaleFactor,
                                                                    minNeighbors=minNeighbors)

            all_boxes = all_boxes + boxes
            all_confidences = all_confidences + confidences
            all_times.append(total_time)

            final_frame = frame.copy()
            for box in all_boxes:
                (startX, startY, endX, endY) = box

                cv2.rectangle(final_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            for box in ground_truth_boxes['frame_' + str(current_frame)]:
                (startX, startY, endX, endY) = box

                cv2.rectangle(final_frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            frames_times.append(np.sum(all_times))

    if output_video == 'y':
        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(main_dir + '/results/' + video_file_name + '_haar_prediction_result.mp4', fourcc, 30, (final_frame.shape[1], final_frame.shape[0]), True)

        # write the output frame to disk
        writer.write(final_frame)

    dict_predictions['frame_' + str(current_frame)] = {}
    dict_predictions['frame_' + str(current_frame)]['boxes'] = all_boxes
    dict_predictions['frame_' + str(current_frame)]['scores'] = all_confidences

    current_frame += 1
    progress_bar.update(1)

    ##Debug
    cv2.imshow('frame', final_frame)
    # Press Q on keyboard to  exit
    if (cv2.waitKey(25) & 0xFF == ord('q')) or current_frame == end_frame:
        vs.release()
        cv2.destroyAllWindows()  # Debug
        if output_video == 'y':
            writer.release()
        break

if debug != 'y':
    with open(main_dir + '/results/' + video_file_name + '_haar_predicted_boxes.json', 'w') as fp:
        json.dump(dict_predictions, fp)

    print(np.round(1/np.mean(frames_times), 2), 'FPS')