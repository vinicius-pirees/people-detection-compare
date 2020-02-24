import numpy as np
import cv2
from tqdm import tqdm
import json
import psutil
import time
import os

# current_process = psutil.Process()
# print(current_process.cpu_percent())


def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    """
    Copyright (c) 2015-2016 Adrian Rosebrock, http://www.pyimagesearch.com
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked and its respectives probabilities
    return boxes[pick].astype("int"), np.array(probs)[pick]


def tiles_info(image, tiles_x, tiles_y, margin_percent):
    tiles_dict = {}
    rect_dimension_x = image.shape[0] / tiles_x
    rect_dimension_y = image.shape[1] / tiles_y

    for tile_x in range(0, tiles_x):
        for tile_y in range(0, tiles_y):
            x = int(tile_x * rect_dimension_x)
            y = int(tile_y * rect_dimension_y)
            final_x = int(x + rect_dimension_x)
            final_y = int(y + rect_dimension_y)

            margin_x = int(rect_dimension_x * margin_percent)
            margin_y = int(rect_dimension_y * margin_percent)

            new_x = max(0, x - margin_x)
            new_y = max(0, y - margin_y)
            new_final_x = min(image.shape[0], final_x + margin_x)
            new_final_y = min(image.shape[1], final_y + margin_y)

            margin_left = x - new_x
            margin_top = y - new_y
            margin_right = new_final_x - final_x
            margin_bottom = new_final_y - final_y

            tiles_dict[(tile_x, tile_y)] = {
                'coords': (x, y, final_x, final_y),
                'new_coords': (new_x, new_y, new_final_x, new_final_y),
                'margins_length': (margin_left, margin_top, margin_right, margin_bottom)}
    return tiles_dict


def tile_image(image, tiles_x, tiles_y, tiles_dict):

    tiles = np.zeros((tiles_x, tiles_y), object)

    for tile_x in range(0, tiles_x):
        for tile_y in range(0, tiles_y):
            (x, y, final_x, final_y) = tiles_dict[tile_x, tile_y]['new_coords']
            tile = image[x:final_x, y:final_y]
            tiles[tile_x, tile_y] = tile

    return tiles


def append_tiles_image(tiles, tiles_x, tiles_y):
    horizontal_tiles = []
    horizontal_img = None
    for line in range(0, tiles_x):
        for column in range(0, tiles_y):
            if column == 0:
                horizontal_img = tiles[line, column].copy()
            else:
                horizontal_img = np.concatenate((horizontal_img, tiles[line, column].copy()), axis=1)
        horizontal_tiles.append(horizontal_img)

    final_img = None
    horizontal_array = np.array(horizontal_tiles)
    for line in range(0, tiles_x):
        if line == 0:
            final_img = horizontal_array[line].copy()
        else:
            final_img = np.concatenate((final_img, horizontal_array[line].copy()), axis=0)

    return final_img


def box_new_coords(box, row, column, tiles_dict):
    (startX, startY, endX, endY) = box

    new_x, new_y, new_final_x, new_final_y = tiles_dict[row, column]['new_coords']

    startX += new_y
    startY += new_x
    endX += new_y
    endY += new_x

    return startX, startY, endX, endY


def detect_over_frames(video_dict, technique, detect_single_frame_function, **kwargs):
    output_video = kwargs.get('output_video')
    detect_on_tiles = kwargs.get('detect_on_tiles')
    debug = kwargs.get('debug')
    tiles_x = kwargs.get('tiles_x')
    tiles_y = kwargs.get('tiles_y')
    main_dir = kwargs.get('main_dir')
    moving_frames = kwargs.get('moving_frames')

    total_frames_all_videos = kwargs.get('total_frames_all_videos')

    model = kwargs.get('model')

    ground_truth_boxes = kwargs.get('ground_truth_boxes')

    detection_args = {}
    if technique == 'haar':
        detection_args['scaleFactor'] = kwargs.get('scaleFactor')
        detection_args['minNeighbors'] = kwargs.get('minNeighbors')
    if technique == 'haar_confidence':
        detection_args['scaleFactor'] = kwargs.get('scaleFactor')
        detection_args['minNeighbors'] = kwargs.get('minNeighbors')
    elif technique == 'hog':
        detection_args['winStride'] = kwargs.get('winStride')
        detection_args['padding'] = kwargs.get('padding')
        detection_args['scale'] = kwargs.get('scale')
    elif technique == 'hog_confidence':
        detection_args['winStride'] = kwargs.get('winStride')
        detection_args['padding'] = kwargs.get('padding')
        detection_args['scale'] = kwargs.get('scale')
    elif technique == 'hog_nms':
        detection_args['winStride'] = kwargs.get('winStride')
        detection_args['padding'] = kwargs.get('padding')
        detection_args['scale'] = kwargs.get('scale')
    elif technique == 'yolo':
        detection_args['confidence'] = kwargs.get('confidence')
        detection_args['threshold'] = kwargs.get('threshold')
    elif technique == 'yolo_darknet':
        detection_args['confidence'] = kwargs.get('confidence')
        detection_args['threshold'] = kwargs.get('threshold')
    elif technique == 'mobile_ssd':
        detection_args['confidence'] = kwargs.get('confidence')

    writer = None
    dict_predictions = {}
    progress_bar = tqdm(total=total_frames_all_videos)
    frames_times = []

    for video in video_dict.keys():

        start_frame = video_dict[video].get('start_frame')
        end_frame = video_dict[video].get('end_frame')
        tiles_dict = video_dict[video].get('tiles_dict')
        video_file_path = video_dict[video].get('video_file_path')
        current_frame = start_frame

        video_stream = cv2.VideoCapture(video_file_path)
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Read from start frame

        while True:
            success, frame = video_stream.read()
            if not success:
                video_stream.release()
                cv2.destroyAllWindows()
                if output_video == 'y':
                    writer.release()
                break

            if moving_frames is not None:
                condition = moving_frames[video][str(current_frame)]
                if not condition:
                    current_frame += 1
                    progress_bar.update(1)
                    continue

            all_boxes = []
            all_confidences = []
            all_times = []

            if detect_on_tiles == 'y':
                tiles = tile_image(frame, tiles_x, tiles_y, tiles_dict)

                if debug == 'y':
                    # Debug a specific tile
                    row = 1
                    column = 1
                    boxes, confidences, total_time, final_frame = \
                        detect_single_frame_function(model, tiles[row, column], None,  None,  None,  **detection_args)

                else:
                    final_frame = frame.copy()

                    boxes, confidences, total_time, _ = \
                        detect_single_frame_function(model, frame, None, None, None, **detection_args)

                    all_boxes = all_boxes + boxes
                    all_confidences = all_confidences + confidences
                    all_times.append(total_time)

                    for row in range(0, tiles_x):
                        for column in range(0, tiles_y):
                            boxes, confidences, total_time, _ = \
                                detect_single_frame_function(model, tiles[row, column], row, column, tiles_dict, **detection_args)

                            all_boxes = all_boxes + boxes
                            all_confidences = all_confidences + confidences
                            all_times.append(total_time)

                    for box in all_boxes:
                        (startX, startY, endX, endY) = box

                        cv2.rectangle(final_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                    # for box in ground_truth_boxes[video + '_frame_' + str(current_frame)]:
                    #     (startX, startY, endX, endY) = box
                    #
                    #     cv2.rectangle(final_frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

                    frames_times.append(np.sum(all_times))

            else:  # See entire frame
                if debug == 'y':
                    boxes, confidences, total_time, final_frame = \
                        detect_single_frame_function(model, frame, None, None, None, **detection_args)
                else:
                    boxes, confidences, total_time, _ = \
                        detect_single_frame_function(model, frame,  None,  None, None, **detection_args)

                    all_boxes = all_boxes + boxes
                    all_confidences = all_confidences + confidences
                    all_times.append(total_time)

                    final_frame = frame.copy()
                    for box in all_boxes:
                        (startX, startY, endX, endY) = box

                        cv2.rectangle(final_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                    # for box in ground_truth_boxes[video + '_frame_' + str(current_frame)]:
                    #     (startX, startY, endX, endY) = box
                    #
                    #     cv2.rectangle(final_frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                    frames_times.append(np.sum(all_times))

            if output_video == 'y':
                # check if the video writer is None
                if writer is None:
                    # initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    if detect_on_tiles == 'y':
                        tile_name = '_tiled_' + str(tiles_x) + 'X' + str(tiles_y)
                        file_name_record = main_dir + '/results/VIRAT_videos_' + technique + tile_name + '_prediction_result.mp4'
                    else:
                        file_name_record = main_dir + '/results/VIRAT_videos__' + technique + '_prediction_result.mp4'
                    writer = cv2.VideoWriter(file_name_record,
                                             fourcc, 30, (final_frame.shape[1], final_frame.shape[0]), True)

                # write the output frame to disk
                writer.write(final_frame)

            dict_predictions[video + '_frame_' + str(current_frame)] = {}
            dict_predictions[video + '_frame_' + str(current_frame)]['boxes'] = all_boxes
            dict_predictions[video + '_frame_' + str(current_frame)]['scores'] = all_confidences

            current_frame += 1
            progress_bar.update(1)

            ##Debug
            # cv2.imshow('frame', final_frame)
            # # Press Q on keyboard to  exit
            # if (cv2.waitKey(25) & 0xFF == ord('q')) or current_frame == end_frame:
            #     video_stream.release()
            #     cv2.destroyAllWindows()  # Debug
            #     if output_video == 'y':
            #         writer.release()
            #     break

    if debug != 'y':
        if detect_on_tiles == 'y':
            if moving_frames is not None:
                tile_name = '_tiled_' + str(tiles_x) + 'X' + str(tiles_y) + '_moving'
            else:
                tile_name = '_tiled_' + str(tiles_x) + 'X' + str(tiles_y)
            prediction_file_name = main_dir + '/results/VIRAT_videos_' + technique + tile_name + '_predicted_boxes.json'
        else:
            if moving_frames is not None:
                prediction_file_name = main_dir + '/results/VIRAT_videos_' + technique + '_moving_predicted_boxes.json'
            else:
                prediction_file_name = main_dir + '/results/VIRAT_videos_' + technique + '_predicted_boxes.json'

        with open(prediction_file_name, 'w') as fp:
            json.dump(dict_predictions, fp)

        # Compute FPS
        if detect_on_tiles == 'y':
            tile_name = '_tiled_' + str(tiles_x) + 'X' + str(tiles_y)
            fps_file = main_dir + '/results/VIRAT_videos_' + technique + tile_name + '_FPS.txt'
        else:
            fps_file = main_dir + '/results/VIRAT_videos_' + technique + '_FPS.txt'
        fps = np.round(1 / np.mean(frames_times), 2)
        with open(fps_file, 'w') as fp:
            fp.write(str(fps))


def get_cpu_avg_usage(pid):
    if os.path.exists("/tmp/temp_file.txt"):
        os.remove("/tmp/temp_file.txt")
    cpu_usage_percents = []
    process = psutil.Process(pid=pid)
    while True:
        try:
            percent = process.cpu_percent()
            cpu_usage_percents.append(percent)
            with open('/tmp/temp_file.txt', 'a') as f:
                f.write("%s\n" % percent)
        except Exception as e:
            print(e)
            break
        time.sleep(1)
    return float(np.mean(cpu_usage_percents))


def write_cpu_usage_file(main_dir, video_file_name, technique, tiles):
    if tiles is not None:
        tiles_x = tiles[0]
        tiles_y = tiles[2]
        tile_name = '_tiled_' + tiles_x + 'X' + tiles_y
        cpu_file = main_dir + '/results/' + video_file_name + '_' + technique + tile_name + '_AVG_CPU.txt'
    else:
        cpu_file = main_dir + '/results/' + video_file_name + '_' + technique + '_AVG_CPU.txt'

    numbers = np.loadtxt('/tmp/temp_file.txt')
    with open(cpu_file, 'w') as fp:
        fp.write(str(float(np.mean(numbers))))

