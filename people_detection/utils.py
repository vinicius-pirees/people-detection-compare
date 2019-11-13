import numpy as np


def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    """
    Copyright (c) 2015-2016 Adrian Rosebrock, http://www.pyimagesearch.com
    """
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

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

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")


def tile_image_old(image, tiles_x, tiles_y):
    rect_dimension_x = image.shape[0] / tiles_x
    rect_dimension_y = image.shape[1] / tiles_y

    tiles = np.zeros((tiles_x, tiles_y), object)

    for tile_x in range(0, tiles_x):
        for tile_y in range(0, tiles_y):
            x = int(tile_x * rect_dimension_x)
            y = int(tile_y * rect_dimension_y)
            final_x = int(x + rect_dimension_x)
            final_y = int(y + rect_dimension_y)

            tile = image[x:final_x, y:final_y]
            tiles[tile_x, tile_y] = tile

    return tiles


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


def tile_new_coords(box, tile_info, margin_info=None):
    (startX, startY, endX, endY) = box

    startX += tile_info['column'] * int(tile_info['dim_y'])
    startY += tile_info['row'] * int(tile_info['dim_x'])
    endX += tile_info['column'] * int(tile_info['dim_y'])
    endY += tile_info['row'] * int(tile_info['dim_x'])

    if margin_info is not None:
        tile_start_x = margin_info['margin_left']
        tile_end_x = tile_start_x + tile_info['dim_x']
        tile_start_y = margin_info['margin_top']
        tile_end_y = tile_start_y + tile_info['dim_y']

        if startX > tile_end_x or startX < tile_start_x:
            if startX > tile_end_x:
                delta = startX - tile_end_x
                startX += delta
                endX += delta
            else:
                delta = startX - tile_start_x
                startX += delta
                endX += delta

        if startY > tile_end_y or startY < tile_start_y:
            if startY > tile_end_y:
                delta = startY - tile_end_y
                startY += delta
                endY += delta
            else:
                delta = startY - tile_start_y
                startY += delta
                endY += delta

        return startX, startY, endX, endY
