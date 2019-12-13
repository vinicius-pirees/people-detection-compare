import cv2
import os, sys
import numpy as np
import time
from matplotlib import pyplot as plt
from utils import tile_image, append_tiles_image, tiles_info, box_new_coords, detect_over_frames
from tqdm import tqdm
import json
import argparse


# video_name = 'VIRAT_S_010000_00_000000_000165'
main_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

videos_path = '/home/vgoncalves/personal-git/people-detection-compare/resources/virat_dataset/'

video_name_list = ['VIRAT_S_000201_02_000590_000623',
                   'VIRAT_S_010000_00_000000_000165',
                   'VIRAT_S_010003_01_000111_000137',
                   'VIRAT_S_010106_01_000493_000526',
                   'VIRAT_S_010200_03_000470_000567',
                   'VIRAT_S_050000_12_001591_001619']


# start_frame = 550
# end_frame = 570
start_frame = None
end_frame = None

parser = argparse.ArgumentParser(description='Merge')

parser.add_argument('-g', '--ground-truth-boxes',
                    dest='ground_truth_boxes',
                    type=str,
                    default='ground_truth_boxes.json')
parser.add_argument('-p', '--predicted-boxes',
                    dest='predicted_boxes',
                    type=str,
                    default='predicted_boxes.json')
args = parser.parse_args()
ground_truth_boxes = args.ground_truth_boxes
predicted_boxes = args.predicted_boxes

# ground_truth_boxes = '/home/vgoncalves/personal-git/people-detection-compare/resources/virat_dataset/VIRAT_S_010000_00_000000_000165_ground_truth_boxes.json'
# predicted_boxes = '/home/vgoncalves//personal-git/people-detection-compare/results/VIRAT_S_010000_00_000000_000165_mobile_ssd_tiled_3X4_moving_predicted_boxes.json'
# video_file = '/home/vgoncalves/personal-git/people-detection-compare/resources/virat_dataset/VIRAT_S_010000_00_000000_000165.mp4'

with open(ground_truth_boxes) as infile:
    gt_boxes = json.load(infile)

with open(predicted_boxes) as infile:
    pred_boxes = json.load(infile)


for video_name in video_name_list:
    video_file_path = videos_path + video_name + '.mp4'

    vs = cv2.VideoCapture(video_file_path)

    num_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    # if start_frame is None:
    #     start_frame = 0
    # if end_frame is None:
    #     end_frame = num_frames
    start_frame = 0
    end_frame = num_frames

    vs.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Read from start frame
    total_frames = end_frame - start_frame
    current_frame = start_frame

    while True:
        success, frame = vs.read()
        if not success:
            vs.release()
            cv2.destroyAllWindows()
            break

        if pred_boxes.get(video_name + '_frame_' + str(current_frame)):
            for box in pred_boxes[video_name + '_frame_' + str(current_frame)]['boxes']:
                (startX, startY, endX, endY) = box

                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        if gt_boxes.get(video_name + '_frame_' + str(current_frame)):
            for box in gt_boxes[video_name + '_frame_' + str(current_frame)]:
                (startX, startY, endX, endY) = box

                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)

        ##Debug
        cv2.imshow('frame', frame)
        # Press Q on keyboard to  exit
        if (cv2.waitKey(25) & 0xFF == ord('q')) or current_frame == end_frame:
            vs.release()
            cv2.destroyAllWindows()  # Debug
            break
        current_frame += 1
