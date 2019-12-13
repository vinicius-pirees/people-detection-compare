import cv2
import numpy as np
import os
import pandas as pd
import json

main_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
only_moving_frames = 'n'
video_name_list = ['VIRAT_S_000201_02_000590_000623',
                   'VIRAT_S_010000_00_000000_000165',
                   'VIRAT_S_010003_01_000111_000137',
                   'VIRAT_S_010106_01_000493_000526',
                   'VIRAT_S_010200_03_000470_000567',
                   'VIRAT_S_050000_12_001591_001619']
dict_gt_boxes = {}

if only_moving_frames == 'y':
    moving_frames_file = main_dir + '/results/VIRAT_videos_moving_frames.json'
    with open(moving_frames_file) as infile:
        moving_frames = json.load(infile)

for video_name in video_name_list:
    video = cv2.VideoCapture(video_name + '.mp4')

    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize
    if only_moving_frames == 'n':
        for frame in range(0, num_frames):
            dict_gt_boxes[video_name + '_frame_' + str(frame)] = []


    '''
    Each line captures information about a bounding box of an object (person/car etc) at the corresponding frame.
    Each object track is assigned a unique 'object id' identifier.
    Note that:
    - an object may be moving or static (e.g., parked car).
    - an object track may be fragmented into multiple tracks.

    Object File Columns
    1: Object id        (a unique identifier of an object track. Unique within a file.)
    2: Object duration  (duration of the object track)
    3: Currnet frame    (corresponding frame number)
    4: bbox lefttop x   (horizontal x coordinate of the left top of bbox, origin is lefttop of the frame)
    5: bbox lefttop y   (vertical y coordinate of the left top of bbox, origin is lefttop of the frame)
    6: bbox width       (horizontal width of the bbox)
    7: bbox height      (vertical height of the bbox)
    8: Objct Type       (object type)

    Object Type ID (for column 8 above for object files)
    1: person
    2: car              (usually passenger vehicles such as sedan, truck)
    3: vehicles         (vehicles other than usual passenger cars. Examples include construction vehicles)
    4: object           (neither car or person, usually carried objects)
    5: bike, bicylces   (may include engine-powered auto-bikes)
    '''


    annotations = np.loadtxt(video_name + '.viratdata.objects.txt')

    annotation_df = pd.read_csv(video_name + '.viratdata.objects.txt',
                                header=None, sep=' ',
                                names=['objectId',
                                       'objectDuration',
                                       'currentFrame', 'x', 'y', 'width', 'height', 'objectType'])


    annotation_df = annotation_df.sort_values(by=['currentFrame'])

    annotation_df_people = annotation_df[annotation_df['objectType'] == 1]


    if only_moving_frames == 'y':
        for frame in moving_frames[video_name]:
            condition = moving_frames[video_name][str(frame)]
            if condition:
                dict_gt_boxes[video_name + '_frame_' + str(frame)] = []
                for index, row in annotation_df_people[annotation_df_people['currentFrame'] == float(frame)].iterrows():
                    dict_gt_boxes[video_name + '_frame_' + str(frame)].append([int(row['x']),
                                                                 int(row['y']),
                                                                 int(row['x'] + row['width']),
                                                                 int(row['y'] + row['height'])])
    else:
        for frame in range(0, num_frames):
            for index, row in annotation_df_people[annotation_df_people['currentFrame'] == frame].iterrows():
                if dict_gt_boxes.get(video_name + '_frame_' + str(frame)) is None:
                    dict_gt_boxes[video_name + '_frame_' + str(frame)] = []
                dict_gt_boxes[video_name + '_frame_' + str(frame)].append([int(row['x']),
                                                             int(row['y']),
                                                             int(row['x'] + row['width']),

                                                             int(row['y'] + row['height'])])
    if only_moving_frames == 'y':
        with open('VIRAT_videos_moving_ground_truth_boxes.json', 'w') as fp:
            json.dump(dict_gt_boxes, fp)
    else:
        with open('VIRAT_videos_ground_truth_boxes.json', 'w') as fp:
            json.dump(dict_gt_boxes, fp)
