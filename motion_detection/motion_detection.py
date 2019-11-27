import cv2
import os
import numpy as np
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import json

min_area = 500
# start_frame = 550
# end_frame = 600
start_frame = None
end_frame = None
weight = 0.6

video_file = '/home/vgoncalves/personal-git/people-detection-compare/resources/virat_dataset/VIRAT_S_010000_00_000000_000165.mp4'

video_file_name = os.path.splitext(os.path.split(video_file)[1])[0]
main_dir = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

vs = cv2.VideoCapture(video_file)
num_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))

if start_frame is None:
    start_frame = 0
if end_frame is None:
    end_frame = num_frames

vs.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Read from start frame

total_frames = end_frame - start_frame
current_frame = start_frame


def detect_movement_frame(frame, avg, weight, min_area):
    frame_copy = frame.copy()
    frame_gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, tuple([21, 21]), 0)

    if avg is None:
        avg = frame_gray.copy().astype("float")

    frameDelta = cv2.absdiff(frame_gray, cv2.convertScaleAbs(avg))
    cv2.accumulateWeighted(frame_gray, avg, weight)

    thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in contours:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame_copy, avg


def is_current_frame_moving(frame, avg, weight, min_area):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, tuple([21, 21]), 0)

    if avg is None:
        avg = frame_gray.copy().astype("float")

    frameDelta = cv2.absdiff(frame_gray, cv2.convertScaleAbs(avg))
    cv2.accumulateWeighted(frame_gray, avg, weight)

    thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in contours:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) >= min_area:
            return True, avg

    return False, avg


if __name__ == "__main__":
    progress_bar = tqdm(total=total_frames)
    avg = None
    moving_frames = {}

    while True:
        success, frame = vs.read()

        if not success:
            vs.release()
            cv2.destroyAllWindows()
            break
        frame_copy = frame.copy()

        is_moving, avg = is_current_frame_moving(frame, avg, weight, min_area)
        moving_frames[current_frame] = is_moving

        text = str(is_moving)
        cv2.putText(frame_copy, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.imshow('Moving Status', frame_copy)

        frame_with_boxes, _ = detect_movement_frame(frame, avg, weight, min_area)
        cv2.imshow('Bounding Boxes', frame_with_boxes)

        # Press Q on keyboard to  exit
        if (cv2.waitKey(25) & 0xFF == ord('q')) or current_frame == end_frame:
            vs.release()
            cv2.destroyAllWindows()
            break

        current_frame += 1
        progress_bar.update(1)

    with open(main_dir + '/results/' + video_file_name + '_moving_frames.json', 'w') as fp:
        json.dump(moving_frames, fp)
