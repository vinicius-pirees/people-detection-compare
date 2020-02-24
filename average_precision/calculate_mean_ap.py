"""
author: Timothy C. Arlen
date: 28 Feb 2018

Calculate Mean Average Precision (mAP) for a set of bounding boxes corresponding to specific
image Ids. Usage:

> python calculate_mean_ap.py

"""

from __future__ import absolute_import, division, print_function

from copy import deepcopy
import json
import glob
import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle

main_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
#
# sns.set_style('white')
# sns.set_context('poster')




# COLORS = [
#     '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
#     '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
#     '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
#     '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']


COLORS = [
    '#bada55','#ff0000','#ff80ed','#5ac18e','#ffa500','#660066','#c7c7c7','#17becf']

scales = {'small': [0, np.power(32, 2)],
         'medium': [np.power(32, 2), np.power(96, 2)],
         'large': [np.power(96, 2), np.infty]}


def box_area(box):
    x1, y1, x2, y2 = box
    
    if (x1 > x2) or (y1 > y2):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
        
    box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    return box_area

def filter_boxes(boxes, scale='all'):
    if scale != 'all':
        filtered_boxes = []
        for box in boxes:
            area = box_area(box)
            if area >= scales[scale][0] and area < scales[scale][1]:
                filtered_boxes.append(box)
            
        return filtered_boxes
    else:
        return boxes


def box_in_scale(box, scale='all'):
    if scale != 'all':
        area = box_area(box)
        if scales[scale][0] <= area < scales[scale][1]:
            return True
        else:
            return False
    else:
        return True

def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box

    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

    Returns:
        float: value of the IoU for the two boxes.

    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def get_single_image_results(gt_boxes, pred_boxes, iou_thr, scale='all'):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))
    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    if len(all_gt_indices) == 0:
        tp = 0
        # Consider only unmatched detections inside of area range
        pred_boxes = filter_boxes(pred_boxes, scale=scale)
        fp = len(pred_boxes)
        fn = 0
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        # Consider only unmatched detections inside of area range
        pred_boxes = filter_boxes(pred_boxes, scale=scale)
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)

        # Consider only unmatched detections inside of area range
        num_unmatched_pred_boxes_ignore = 0
        idx_unmatched_pre_boxes = set(all_pred_indices) - set(pred_match_idx)
        for idx_pred in idx_unmatched_pre_boxes:
            if not box_in_scale(pred_boxes[idx_pred], scale=scale):
                num_unmatched_pred_boxes_ignore += 1

        tp = len(gt_match_idx)
        fp = (len(pred_boxes) - len(pred_match_idx)) - num_unmatched_pred_boxes_ignore
        assert fp >= 0
        fn = len(gt_boxes) - len(gt_match_idx)

    return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}


def calc_precision_recall(img_results):
    """Calculates precision and recall from the set of images

    Args:
        img_results (dict): dictionary formatted like:
            {
                'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                'img_id2': ...
                ...
            }

    Returns:
        tuple: of floats of (precision, recall)
    """
    true_pos = 0; false_pos = 0; false_neg = 0
    for _, res in img_results.items():
        true_pos += res['true_pos']
        false_pos += res['false_pos']
        false_neg += res['false_neg']

    try:
        precision = true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        recall = 0.0

    return (precision, recall)

def get_model_scores_map(pred_boxes):
    """Creates a dictionary of from model_scores to image ids.

    Args:
        pred_boxes (dict): dict of dicts of 'boxes' and 'scores'

    Returns:
        dict: keys are model_scores and values are image ids (usually filenames)

    """
    model_scores_map = {}
    for img_id, val in pred_boxes.items():
        for score in val['scores']:
            if score not in model_scores_map.keys():
                model_scores_map[score] = [img_id]
            else:
                model_scores_map[score].append(img_id)
    return model_scores_map

def get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=0.5, scores_all_same=False, scale='all', use_pickle=False, name=None):
    """Calculates average precision at given IoU threshold.

    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (list of list of floats): list of locations of predicted
            objects as [xmin, ymin, xmax, ymax]
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
        scale (string) one of 'all','small','medium','large'. The scale of interest 
            for calculating precision and recall

    Returns:
        dict: avg precision as well as summary info about the PR curve

        Keys:
            'avg_prec' (float): average precision for this IoU threshold
            'precisions' (list of floats): precision value for the given
                model_threshold
            'recall' (list of floats): recall value for given
                model_threshold
            'models_thrs' (list of floats): model threshold value that
                precision and recall were computed for.
    """
    assert scale in ['all', 'small', 'medium', 'large']

    num_gt_boxes = 0
    for img in gt_boxes.keys():
        boxes = gt_boxes[img]
        gt_boxes[img] = []
        for box in boxes:
            if box_in_scale(box, scale=scale):
                gt_boxes[img].append(box)
                num_gt_boxes += 1

    print("Number of", scale, "ground truth boxes:", num_gt_boxes)

    # Normalization and rounding
    all_scores = []
    for img in pred_boxes.keys():
        for score in pred_boxes[img]['scores']:
            all_scores.append(score)

    min_score = min(all_scores)
    max_score = max(all_scores)

    for img in pred_boxes.keys():
        scores_temp = pred_boxes[img]['scores']
        pred_boxes[img]['scores'] = []
        for score in scores_temp:
            pred_boxes[img]['scores'].append(np.round((score - min_score) / (max_score - min_score), 3))



    model_scores_map = get_model_scores_map(pred_boxes)
    sorted_model_scores = sorted(model_scores_map.keys())

    if use_pickle:
        dict_pickle = pickle.load(open(main_dir + '/average_precision/recall_precision_' + name + '_' + scale + '.pickle', 'rb'))
        precisions = np.array(dict_pickle['precision'])
        # precisions_test = np.sort(precisions.copy())[::-1]
        # print(precisions_test[0:100])
        recalls = np.array(dict_pickle['recall'])
        # recalls_test = np.sort(recalls.copy())[::-1]
        # print(recalls_test[0:100])
        prec_at_rec = []
    else:

        # Sort the predicted boxes in descending order (lowest scoring boxes first):
        for img_id in pred_boxes.keys():
            arg_sort = np.argsort(pred_boxes[img_id]['scores'])
            pred_boxes[img_id]['scores'] = np.array(pred_boxes[img_id]['scores'])[arg_sort].tolist()
            pred_boxes[img_id]['boxes'] = np.array(pred_boxes[img_id]['boxes'])[arg_sort].tolist()

        pred_boxes_pruned = deepcopy(pred_boxes)

        precisions = []
        recalls = []
        model_thrs = []
        img_results = {}

        if not scores_all_same:
            sorted_models_scores_list = sorted_model_scores[:-1]
        else:
            sorted_models_scores_list = sorted_model_scores

        progress_bar = tqdm(total=len(sorted_models_scores_list))
        # Loop over model score thresholds and calculate precision, recall
        for ithr, model_score_thr in enumerate(sorted_models_scores_list):
            # On first iteration, define img_results for the first time:
            img_ids = gt_boxes.keys() if ithr == 0 else model_scores_map[model_score_thr]
            for img_id in img_ids:
                gt_boxes_img = gt_boxes[img_id]
                box_scores = pred_boxes_pruned[img_id]['scores']
                start_idx = 0
                for score in box_scores:
                    if score <= model_score_thr:
                        pred_boxes_pruned[img_id]
                        start_idx += 1
                    else:
                        break

                if not scores_all_same:
                    # Remove boxes, scores of lower than threshold scores:
                    pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
                    pred_boxes_pruned[img_id]['boxes'] = pred_boxes_pruned[img_id]['boxes'][start_idx:]

                # Recalculate image results for this image
                img_results[img_id] = get_single_image_results(
                    gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr, scale=scale)

            prec, rec = calc_precision_recall(img_results)

            precisions.append(prec)
            recalls.append(rec)
            #model_thrs.append(model_score_thr)
            progress_bar.update(1)

        dict_pickle = {'precision': precisions, 'recall': recalls}
        pickle.dump(dict_pickle, open('recall_precision_' + name + '_' + scale + '.pickle', 'wb'))

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        prec_at_rec = []

    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    # return {
    #     'avg_prec': avg_prec,
    #     'precisions': precisions,
    #     'recalls': recalls,
    #     'model_thrs': model_thrs}

    return {
        'avg_prec': avg_prec,
        'precisions': precisions,
        'recalls': recalls}




def simple_precision_recall(gt_boxes, pred_boxes, iou_thr=0.5, scale='all'):
    img_results = {}
    img_ids = gt_boxes.keys()

    for img_id in img_ids:
        gt_boxes_img = gt_boxes[img_id]
        box_scores = pred_boxes[img_id]['scores']

        # Recalculate image results for this image
        img_results[img_id] = get_single_image_results(
            gt_boxes_img, pred_boxes[img_id]['boxes'], iou_thr, scale=scale)

    precision, recall = calc_precision_recall(img_results)
    return precision, recall


def plot_pr_curve(
    precisions, recalls, title='Precision-Recall curve for Person', label=None, color=None, ax=None):
    """Simple plotting helper function"""

    if ax is None:
        plt.figure(figsize=(10,8))
        ax = plt.gca()

    if color is None:
        color = COLORS[0]
    #ax.scatter(recalls, precisions, label=label, s=10, color=color)
    ax.plot(recalls, precisions,  label=label,  linewidth=3, markersize=1, color=color)
    ax.set_xlabel(r'$Recall$', fontsize=15)
    ax.set_ylabel(r'$Precision$', fontsize=15)
    #ax.set_title(title)
    ax.set_xlim([0.0,1.1])
    ax.set_ylim([0.0,1.1])
    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge')

    parser.add_argument('-g', '--ground-truth-boxes',
                        dest='ground_truth_boxes',
                        type=str,
                        default='ground_truth_boxes.json')
    parser.add_argument('-p', '--predicted-boxes',
                        dest='predicted_boxes',
                        type=str,
                        default='predicted_boxes.json')
    parser.add_argument('-s', '--scores-all-same',
                        dest='scores_all_same',
                        type=str,
                        default='n')
    parser.add_argument('-r', '--scale',
                        dest='scale',
                        type=str,
                        default='all')
    parser.add_argument('-u', '--use-pickle',
                        dest='use_pickle',
                        type=str,
                        default='n')
    parser.add_argument('-n', '--name',
                        dest='name',
                        type=str)

    args = parser.parse_args()
    ground_truth_boxes = args.ground_truth_boxes
    predicted_boxes = args.predicted_boxes
    scores_all_same = args.scores_all_same
    scale = args.scale
    use_pickle = args.use_pickle
    name = args.name



    with open(ground_truth_boxes) as infile:
        gt_boxes = json.load(infile)

    with open(predicted_boxes) as infile:
        pred_boxes = json.load(infile)

    if scores_all_same == 'y':
        scores_all_same = True
    else:
        scores_all_same = False

    if use_pickle == 'y':
        use_pickle = True
    else:
        use_pickle = False

    start_time = time.time()
    ax = None
    avg_precs = []
    iou_thrs = []
    # for idx, iou_thr in enumerate(np.linspace(0.5, 0.95, 10)):
    #     data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
    #     avg_precs.append(data['avg_prec'])
    #     iou_thrs.append(iou_thr)
    #
    #     precisions = data['precisions']
    #     recalls = data['recalls']
    #     ax = plot_pr_curve(
    #         precisions, recalls, label='{:.2f}'.format(iou_thr), color=COLORS[idx*2], ax=ax)

    iou_thr = 0.5
    data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr, scores_all_same=scores_all_same,
                                    scale=scale, name=name, use_pickle=use_pickle)
    avg_precs.append(data['avg_prec'])
    iou_thrs.append(iou_thr)

    precisions = data['precisions']
    recalls = data['recalls']
    ax = plot_pr_curve(precisions, recalls, label='{:.2f}'.format(iou_thr), ax=ax)

    # prettify for printing:
    avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
    iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
    print('map: {:.2f}'.format(100*np.mean(avg_precs)))
    print('avg precs: ', avg_precs)
    print('iou_thrs:  ', iou_thrs)
    plt.legend(loc='upper right', title='IOU Thr', frameon=True)
    for xval in np.linspace(0.0, 1.0, 11):
        plt.vlines(xval, 0.0, 1.1, color='gray', alpha=0.3, linestyles='dashed')
    end_time = time.time()
    print('\nPlotting and calculating mAP takes {:.4f} secs'.format(end_time - start_time))
    plt.show()