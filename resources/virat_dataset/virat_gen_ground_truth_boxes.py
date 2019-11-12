import cv2
import numpy as np
import os
import pandas as pd
import json


video_name = 'VIRAT_S_010000_00_000000_000165'

video = cv2.VideoCapture(video_name + '.mp4')

num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

dict_gt_boxes = {}
for frame in range(0, num_frames):
    dict_gt_boxes['frame_' + str(frame)] = []

# Each line captures informabiont about a bounding box of an object (person/car etc) at the corresponding frame.
# Each object track is assigned a unique 'object id' identifier.
# Note that:
# - an object may be moving or static (e.g., parked car).
# - an object track may be fragmented into multiple tracks.

# Object File Columns
# 1: Object id        (a unique identifier of an object track. Unique within a file.)
# 2: Object duration  (duration of the object track)
# 3: Currnet frame    (corresponding frame number)
# 4: bbox lefttop x   (horizontal x coordinate of the left top of bbox, origin is lefttop of the frame)
# 5: bbox lefttop y   (vertical y coordinate of the left top of bbox, origin is lefttop of the frame)
# 6: bbox width       (horizontal width of the bbox)
# 7: bbox height      (vertical height of the bbox)
# 8: Objct Type       (object type)

# Object Type ID (for column 8 above for object files)
# 1: person
# 2: car              (usually passenger vehicles such as sedan, truck)
# 3: vehicles         (vehicles other than usual passenger cars. Examples include construction vehicles)
# 4: object           (neither car or person, usually carried objects)
# 5: bike, bicylces   (may include engine-powered auto-bikes)


annotations = np.loadtxt(video_name + '.viratdata.objects.txt')

annotation_df = pd.read_csv(video_name + '.viratdata.objects.txt',
                            header=None, sep=' ',
                            names=['objectId',
                                   'objectDuration',
                                   'currentFrame', 'x', 'y', 'width', 'height', 'objectType'])


annotation_df = annotation_df.sort_values(by=['currentFrame'])

annotation_df_people = annotation_df[annotation_df['objectType'] == 1]

for frame in range(0, num_frames):
    for index, row in annotation_df_people[annotation_df_people['currentFrame'] == frame].iterrows():
        dict_gt_boxes['frame_' + str(frame)].append([int(row['x']),
                                                     int(row['y']),
                                                     int(row['x'] + row['width']),
                                                     int(row['y'] + row['height'])])


with open(video_name + '_gt_boxes.json', 'w') as fp:
    json.dump(dict_gt_boxes, fp)
