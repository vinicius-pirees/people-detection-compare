import cv2
import os
import numpy as np
import time
from matplotlib import pyplot as plt
from utils import tile_image, append_tiles_image, tiles_info, box_new_coords


detect_on_tiles = 'y'
debug = 'y'
tiles_x = 3
tiles_y = 4
video_file = '/home/vgoncalves/personal-git/people_detection_compare/resources/virat_dataset/VIRAT_S_010000_00_000000_000165.mp4'
output_video = 'n'
confidence = 0.2
margin_percent = 0.25

video_file_name = os.path.splitext(os.path.split(video_file)[1])[0]


main_dir = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

mobile_ssd_dir = os.path.join(main_dir, 'models/mobile_ssd')
prototxt = os.path.join(mobile_ssd_dir, 'MobileNetSSD_deploy.prototxt.txt')
model = os.path.join(mobile_ssd_dir, 'MobileNetSSD_deploy.caffemodel')

classes_path = os.path.sep.join([mobile_ssd_dir, "classes"])
MOBILE_SSD_CLASSES = open(classes_path).read().strip().split("\n")

mobile_ssd_net = cv2.dnn.readNetFromCaffe(prototxt, model)


def detect_single_frame(model, frame, confidence=0.2, tile_info=None, tiles_dict=None):
    (h, w) = frame.shape[:2]
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

                if tile_info is not None:
                    startX, startY, endX, endY = box_new_coords(box.astype("int"),
                                                                tile_info['row'],
                                                                tile_info['column'],
                                                                tiles_dict)
                else:
                    (startX, startY, endX, endY) = box.astype("int")


                boxes.append([startX, startY, endX, endY])
                confidences.append(_confidence)

                label = "{:.2f}%".format(_confidence * 100)
                cv2.rectangle(frame_with_boxes, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame_with_boxes, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return boxes, confidences, total_time, frame_with_boxes


vs = cv2.VideoCapture(video_file)  # Get frame size
success, frame = vs.read()
tiles_dict = None
if detect_on_tiles == 'y':
    tiles_dict = tiles_info(frame, tiles_x, tiles_y, margin_percent)

vs = cv2.VideoCapture(video_file) # Video
##read from specific frame
vs.set(cv2.CAP_PROP_POS_FRAMES, 550)
writer = None


while True:
    success, frame = vs.read()

    if not success:
        vs.release()
        cv2.destroyAllWindows() # Debug
        if output_video == 'y':
            writer.release()
        break

    if output_video == 'y' and writer is None:
        pass

    if detect_on_tiles == 'y':
        tiles = tile_image(frame, tiles_x, tiles_y, tiles_dict)
        new_tiles = np.zeros((tiles_x, tiles_y), object)

        tile_dim_x = tiles[0, 0].shape[0]
        tile_dim_y = tiles[0, 0].shape[1]
        all_boxes = []
        all_confidences = []
        all_times = []

        for row in range(0, tiles_x):
            for column in range(0, tiles_y):
                boxes, confidences, total_time, final_frame = detect_single_frame(mobile_ssd_net,
                                                                                  tiles[row, column],
                                                                                  confidence=confidence,
                                                                                  tile_info={
                                                                                      'row': row,
                                                                                      'column': column},
                                                                                  tiles_dict=tiles_dict)

                all_boxes = all_boxes + boxes
                all_confidences = all_confidences + confidences
                all_times.append(total_time)

        final_frame = frame.copy()
        for box in all_boxes:
            (startX, startY, endX, endY) = box

            cv2.rectangle(final_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    else:
        # Debug a specif tile
        if debug == 'y':
            tiles_dict = tiles_info(frame, tiles_x, tiles_y, margin_percent)
            tiles = tile_image(frame, tiles_x, tiles_y, tiles_dict)
            boxes, confidences, total_time, final_frame = detect_single_frame(mobile_ssd_net,
                                                                              tiles[2, 1],
                                                                              confidence=confidence
                                                                              )
        else: # See entire frame
            boxes, confidences, total_time, final_frame = detect_single_frame(mobile_ssd_net,
                                                                              frame,
                                                                              confidence=confidence
                                                                              )

    if output_video == 'y':
        pass

    ##Debug
    cv2.imshow('frame', final_frame)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        vs.release()
        cv2.destroyAllWindows()  # Debug
        if output_video == 'y':
            writer.release()
        break









