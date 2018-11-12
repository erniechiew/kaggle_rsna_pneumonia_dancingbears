import pandas as pd
import numpy as np
import pydicom

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# helper function to calculate IoU
def iou(box1, box2):
    x11, y11, w1, h1 = box1

    if (pd.isnull(x11)):
        return "No ground truth"


    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union



def map_iou(boxes_true, boxes_pred, scores, thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold

    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output:
        map: mean average precision of the image
    """

    # According to the introduction, images with no ground truth bboxes will not be
    # included in the map score unless there is a false positive detection (?)

    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None

    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou == "No ground truth":
                    return miou

                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN

        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m

    return map_total / len(thresholds)




def plot_boxes(patientId, pred_data, valid_data, image_folder = 'data/stage_1_train_images/', img_size = 1024):


    # Generate plot object
    fig, ax = plt.subplots(1)

    dcm_data = pydicom.read_file(image_folder + patientId + '.dcm')
    ax.imshow(dcm_data.pixel_array)

    # Parse prediction string into boxes
    predstring = pred_data.loc[pred_data['patientId'] == patientId]['PredictionString'].iloc[0]

    if (pd.isnull(predstring)):
        return "No predicted boxes"
    else:
        predstring = predstring.strip().split(" ")

        scores = [float(s) for s in predstring[0::5]]
        pred_xs = predstring[1::5]
        pred_ys = predstring[2::5]
        pred_widths = predstring[3::5]
        pred_heights = predstring[4::5]

        for i in range(len(pred_xs)):
            rect = patches.Rectangle((float(pred_xs[i]), float(pred_ys[i])), float(pred_widths[i]), float(pred_heights[i]),
                                     linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)



    # Add ground truth bounding boxes
    bbs = valid_data.loc[valid_data.patientId == patientId, ['x', 'y', 'width', 'height']]

    # Rescale bb sizes (if necessary)
    bbs.x, bbs.y, bbs.width, bbs.height = bbs.x / 1024 * img_size, bbs.y / 1024 * img_size, bbs.width / 1024 * img_size, bbs.height / 1024 * img_size

    # Taken from HK's draw_bbs() code
    for bb in bbs.itertuples():
        rect = patches.Rectangle(
            (bb.x, bb.y), bb.width, bb.height,
            linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)


    return ax



def get_score_for_patientId(patientId, pred_data, valid_data, img_size = 1024):

    predstring = pred_data.loc[pred_data['patientId'] == patientId]['PredictionString'].iloc[0]


    if (pd.isnull(predstring)):
        return "No predicted boxes"
    predstring = predstring.strip().split(" ")

    scores = [float(s) for s in predstring[0::5]]
    pred_xs = predstring[1::5]
    pred_ys = predstring[2::5]
    pred_widths = predstring[3::5]
    pred_heights = predstring[4::5]

    boxes_pred_arr = []
    for i in range(len(pred_xs)):
        boxes_pred = []
        boxes_pred.append(float(pred_xs[i]))
        boxes_pred.append(float(pred_ys[i]))
        boxes_pred.append(float(pred_widths[i]))
        boxes_pred.append(float(pred_heights[i]))

        boxes_pred_arr.append(boxes_pred)

    boxes_pred_arr = np.array(boxes_pred_arr)

    # Add ground truth bounding boxes
    bbs = valid_data.loc[valid_data.patientId == patientId, ['x', 'y', 'width', 'height']]

     # Rescale bbs sizes
    bbs.x, bbs.y, bbs.width, bbs.height = bbs.x / 1024 * img_size, bbs.y / 1024 * img_size, bbs.width / 1024 * img_size, bbs.height / 1024 * img_size

    boxes_true_arr = []
    for bb in bbs.itertuples():
        boxes_true = []
        boxes_true.append(bb.x)
        boxes_true.append(bb.y)
        boxes_true.append(bb.width)
        boxes_true.append(bb.height)

        boxes_true_arr.append(boxes_true)

    boxes_true_arr = np.array(boxes_true_arr)


    return map_iou(boxes_true_arr, boxes_pred_arr, scores = scores)



def remove_small_boxes(predstring, min_dimensions = 3900.):


    if (pd.isnull(predstring)):
        return np.nan

    predstring = predstring.strip().split(" ")

    scores = predstring[0::5]
    pred_xs = predstring[1::5]
    pred_ys = predstring[2::5]
    pred_widths = predstring[3::5]
    pred_heights = predstring[4::5]

    newpredstring = []

    for i in range(len(scores)):
        if np.float(pred_widths[i]) * np.float(pred_heights[i]) >= min_dimensions:
            newpredstring.append(scores[i])
            newpredstring.append(pred_xs[i])
            newpredstring.append(pred_ys[i])
            newpredstring.append(pred_widths[i])
            newpredstring.append(pred_heights[i])

    if len(newpredstring) > 0:
        newpredstring = ' '.join(newpredstring)
        return newpredstring
    else:
        return np.nan




def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def predstring_to_list(predstring):
    if pd.isnull(predstring):
        return np.array([])
    else:

        xs = predstring.strip().split(' ')[1::5]
        ys = predstring.strip().split(' ')[2::5]
        widths = predstring.strip().split(' ')[3::5]
        heights = predstring.strip().split(' ')[4::5]

        confidence_scores = predstring.strip().split(' ')[0::5]
        confidence_scores = [float(ci) for ci in confidence_scores]


        x1 = [float(xi) for xi in xs]
        y1 = [float(yi) for yi in ys]
        x2 = [float(xi) + float(widthi) for (xi, widthi) in zip(xs, widths)]
        y2 = [float(yi) + float(heighti) for (yi, heighti) in zip(ys, heights)]

        boxes_true_arr = []

        for i in range(len(xs)):
            boxes_true = []
            boxes_true.append(x1[i])
            boxes_true.append(y1[i])
            boxes_true.append(x2[i])
            boxes_true.append(y2[i])
            boxes_true.append(confidence_scores[i])

            boxes_true_arr.append(boxes_true)

        return np.array(boxes_true_arr)


def truncate_predstring_based_on_nms(predstring, nms_index):

    if pd.isnull(predstring):
        return np.nan

    predstringlist = predstring.strip().split(' ')

    indices_to_keep = [[i*5, i*5+1, i*5+2, i*5+3, i*5+4] for i in nms_index]
    indices_to_keep = sum(indices_to_keep, [])

    predstringlist = [j for i, j in enumerate(predstringlist) if i in indices_to_keep]

    return ' '.join(predstringlist)


# Remove predicted boxes below specified cutoff
def cutoffer(predstring, cutoff):
    if (len(str(predstring)) < 5):
        return np.nan

    confs = predstring.strip().split(' ')[::5]

    newpredstring = ""

    for i in range(len(confs)):
        if np.float(confs[i]) >= cutoff:
            newpredstring += ' '.join(predstring.split(' ')[5*i:(5*i)+5])
            newpredstring += ' '

    # Remove trailing whitespace
    if (len(newpredstring) > 1):
        if (newpredstring[-1] == ' '):
            newpredstring = newpredstring[:-1]


    if (len(newpredstring) == 0):
        return np.nan
    else:
        return newpredstring


def count_number_of_predicted_boxes(pred_string):
    if pd.isnull(pred_string) or len(pred_string) == 0:
        return 0
    else:
        return len(pred_string.strip().split(' ')) / 5


def sort_predstring_by_confidence(predstring):

    if pd.isnull(predstring) or len(predstring) < 1:
        return ""

    xs = predstring.strip().split(' ')[1::5]
    ys = predstring.strip().split(' ')[2::5]
    widths = predstring.strip().split(' ')[3::5]
    heights = predstring.strip().split(' ')[4::5]

    confidence_scores = predstring.strip().split(' ')[0::5]
    confidence_scores = [float(ci) for ci in confidence_scores]

    confidence_order = np.argsort(confidence_scores)
    confidence_order = confidence_order[::-1]


    predstring = ""

    for index in confidence_order:
        # Append a dummy confidence
        substring = str(confidence_scores[index]) + " " + str(xs[index]) + " " + str(ys[index]) + " " + str(widths[index]) + " " + str(heights[index])

        if len(predstring) > 0:
            predstring += " "
            predstring += substring
        else:
            predstring += substring

    return predstring





def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return ()
    return (x, y, w, h)

def combine_boxes_intersect(boxes):
    if not boxes:
        return []
    noIntersectLoop = False
    noIntersectMain = False
    posIndex = 0
    # keep looping until we have completed a full pass over each rectangle
    # and checked it does not overlap with any other rectangle
    while noIntersectMain == False:
        noIntersectMain = True
        posIndex = 0
        # start with the first rectangle in the list, once the first
        # rectangle has been unioned with every other rectangle,
        # repeat for the second until done
        while posIndex < len(boxes):
            noIntersectLoop = False
            while noIntersectLoop == False and len(boxes) > 1:
                a = boxes[posIndex]
                listBoxes = np.delete(boxes, posIndex, 0)
                index = 0
                for b in listBoxes:
                    #if there is an intersection, the boxes overlap
                    if intersection(a, b):
                        #newBox = union(a,b)
                        newBox = intersection(a, b)
                        listBoxes[index] = newBox
                        boxes = listBoxes
                        noIntersectLoop = False
                        noIntersectMain = False
                        index = index + 1
                        break
                    noIntersectLoop = True
                    index = index + 1
            posIndex = posIndex + 1


    return boxes


def combine_boxes_union(boxes):
    if not boxes:
        return []
    noIntersectLoop = False
    noIntersectMain = False
    posIndex = 0
    # keep looping until we have completed a full pass over each rectangle
    # and checked it does not overlap with any other rectangle
    while noIntersectMain == False:
        noIntersectMain = True
        posIndex = 0
        # start with the first rectangle in the list, once the first
        # rectangle has been unioned with every other rectangle,
        # repeat for the second until done
        while posIndex < len(boxes):
            noIntersectLoop = False
            while noIntersectLoop == False and len(boxes) > 1:
                a = boxes[posIndex]
                listBoxes = np.delete(boxes, posIndex, 0)
                index = 0
                for b in listBoxes:
                    #if there is an intersection, the boxes overlap
                    if intersection(a, b):
                        newBox = union(a,b)
                        #newBox = intersection(a, b)
                        listBoxes[index] = newBox
                        boxes = listBoxes
                        noIntersectLoop = False
                        noIntersectMain = False
                        index = index + 1
                        break
                    noIntersectLoop = True
                    index = index + 1
            posIndex = posIndex + 1


    return boxes


def count_boxes_intersect(boxes, min_count = 3):
    if not boxes:
        return []

    count_results = []

    boxIndex = 0

    while boxIndex < len(boxes):

        count_intersections = 1 # Start with 1 (include the current box as an intersection)

        a = boxes[boxIndex]

        listBoxes = np.delete(boxes, boxIndex, 0)

        for b in listBoxes:
            if intersection(a, b):
                count_intersections = count_intersections + 1

        count_results.append(count_intersections)
        boxIndex = boxIndex + 1

    count_results = np.array(count_results)
    index_to_discard = np.where(count_results < min_count)

    boxes = np.delete(boxes, index_to_discard, 0)

    return [list(box) for box in boxes]



def get_box_list_from_predstring(predstring):

    if pd.isnull(predstring) or len(predstring) < 1:
        return []

    predstring_list = predstring.split(' ')

    xs = predstring.strip().split(' ')[1::5]
    ys = predstring.strip().split(' ')[2::5]
    widths = predstring.strip().split(' ')[3::5]
    heights = predstring.strip().split(' ')[4::5]

    box_list = []
    for i in range(len(xs)):
        boxsubarray = []
        boxsubarray.append(float(xs[i]))
        boxsubarray.append(float(ys[i]))
        boxsubarray.append(float(widths[i]))
        boxsubarray.append(float(heights[i]))
        box_list.append(boxsubarray)


    return box_list


def get_predstring_from_box_list(box_list):
    if len(box_list) == 0:
        return np.nan

    predstring = ""

    for box_set in box_list:
        # Append a dummy confidence
        substring = "0.9" + " " + str(box_set[0]) + " " + str(box_set[1]) + " " + str(box_set[2]) + " " + str(box_set[3])

        if len(predstring) > 0:
            predstring += " "
            predstring += substring
        else:
            predstring += substring

    return predstring



def bounding_box_lr_flip(predstring):
    if pd.isnull(predstring) or len(predstring) < 1:
        return predstring

    xs = predstring.strip().split(' ')[1::5]
    ys = predstring.strip().split(' ')[2::5]
    widths = predstring.strip().split(' ')[3::5]
    heights = predstring.strip().split(' ')[4::5]

    confidence_scores = predstring.strip().split(' ')[0::5]

    new_predstring = ""
    for i in range(len(xs)):

        new_x = 1024. - float(xs[i]) - float(widths[i])

        substring = confidence_scores[i] + " " + str(new_x) + " " + \
          ys[i] + " " + widths[i] + " " + heights[i]

        if len(new_predstring) > 0:
            new_predstring += " "
            new_predstring += substring
        else:
            new_predstring += substring

    return new_predstring
