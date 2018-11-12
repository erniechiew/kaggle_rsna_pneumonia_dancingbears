import tensorflow as tf
import keras
import os
from keras.models import load_model

import pandas as pd
import pickle
from PIL import Image
from tqdm import tqdm
import numpy as np

import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
import json

with open("../SETTINGS.json") as f:
    SETTINGS = json.load(f)

SPLIT_NUMBER = 1

MODEL_WEIGHT_FILE = os.path.join("../", SETTINGS['DETECTION_WEIGHTS_DIR'], "yolo_fold%d.h5" % SPLIT_NUMBER)
TEST_IMG_DIR = os.path.join("../", SETTINGS["TEST_CLEAN_DATA_DIR"])
RESULTS_DIR = os.path.join("../", SETTINGS['DETECTION_RESULTS_DIR'])

ANCHORS = SETTINGS['ANCHOR_SPLIT_%d' % SPLIT_NUMBER]


def generate_predictionstring(image_path, obj_threshold=0.2, nms_threshold=0.45, flip = False):

    image = cv2.imread(image_path)

    if flip:
        image = np.fliplr(image).copy()

    height_, width_, _ = image.shape

    height_ = float(height_)
    width_ = float(width_)

    boxes = get_yolo_boxes(infer_model,
                           [image],
                           608,
                           608,
                           ANCHORS,
                           obj_threshold,
                           nms_threshold)[0]


    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    xmin = [x.xmin for x in boxes]
    xmax = [x.xmax for x in boxes]
    ymin = [x.ymin for x in boxes]
    ymax = [x.ymax for x in boxes]

    widths = [x_max - x_min for x_max, x_min in zip(xmax, xmin)]
    heights = [y_max - y_min for y_max, y_min in zip(ymax, ymin)]


    scores_ = [x.get_score() for x in boxes]

    submission_string = ""


    for i in range(len(xmin)):

        if (i != 0):
            submission_string += " "

        submission_string += str(scores_[i]) + " " + str(round(xmin[i])) + " " + \
          str(round(ymin[i])) + " " + str(round(widths[i])) + " " + str(round(heights[i], 5))



    return submission_string




print("Loading model...")
infer_model = load_model(MODEL_WEIGHT_FILE)
print("Done!")

print("Generating predictions...")
print("Generating predictions on untampered test images...")
test_filenames = os.listdir(TEST_IMG_DIR)
test_filenames = [x.split('.')[0] for x in test_filenames]

submission = pd.DataFrame({'patientId': test_filenames})

pred_strings = []

for test_img_i in tqdm(range(submission.shape[0])):

    imgpath = TEST_IMG_DIR + '/' + submission.iloc[test_img_i]['patientId'] + '.png'
    pred_strings.append(generate_predictionstring(imgpath, 0.2, 0.45))

submission['PredictionString'] = pred_strings

submission.to_csv(os.path.join(RESULTS_DIR,
                                'yolo_fold%d_predictions_on_test.csv' % SPLIT_NUMBER),
                  index = False)

print("Done!")
print("Generating predictions on horizontally flipped test images...")

pred_strings = []

for test_img_i in tqdm(range(submission.shape[0])):

    imgpath = TEST_IMG_DIR + '/' + submission.iloc[test_img_i]['patientId'] + '.png'
    pred_strings.append(generate_predictionstring(imgpath, 0.2, 0.45, True))

submission['PredictionString'] = pred_strings

submission.to_csv(os.path.join(RESULTS_DIR,
                               'yolo_fold%d_predictions_on_test_flipped.csv' % SPLIT_NUMBER),
                  index = False)
print("Done!")
