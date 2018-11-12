# -*- coding: utf-8 -*-
from keras.applications.densenet import DenseNet121
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (
    Input, Dense, Dropout
)
from keras.layers import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate, zoom
from skimage import exposure
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import cv2
import keras
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import json

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

IMG_DIM = [256, 256, 3]
BATCH_SIZE = 32

with open("../SETTINGS.json") as f:
    SETTINGS = json.load(f)

SPLIT_NUMBER = 1

TEST_IMG_DIR = os.path.join("../", SETTINGS["TEST_CLEAN_DATA_DIR"])
WEIGHTS = os.path.join("../", SETTINGS['CLASSIFICATION_WEIGHTS_DIR'], 'classifier_split_%d.h5' % SPLIT_NUMBER)
RESULTS_DIR = os.path.join("../", SETTINGS["CLASSIFICATION_RESULTS_DIR"])

def cv2_clipped_zoom(img, zoom_factor):
    '''
    Center zoom in/out of the given image
    '''
    height, width = img.shape[:2]
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width

    return result


def augmentation(img):
    aug_0 = np.random.random(4)
    aug_0[1] -= 0.2
    aug_0 = aug_0 > 0.5
    aug_1, aug_2, aug_3, aug_4 = aug_0

    img = img.astype(np.uint8)

    if aug_1: img = np.fliplr(img)
    if aug_2:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = img / 255

    if aug_3:
        degree = np.random.uniform(0.8, 1.2)
        img = cv2_clipped_zoom(img, degree)

    if aug_4:
        degree = np.random.uniform(-10, 10)
        img = rotate(img, degree, reshape=False)

    return img


def Generator(files, label, gen_args, flow_args):
    idg = ImageDataGenerator(**gen_args)

    directory = os.path.dirname(files[0])
    gen = idg.flow_from_directory(directory, **flow_args)
    gen.filenames = files
    gen.classes = label
    gen.sample = len(files)
    gen.n = len(files)
    gen._set_index_array()
    gen.directory = ''

    return gen


def roc_auc(y_true, y_pred):
    tf = keras.backend.tf
    value, update_op = tf.metrics.auc(y_true, y_pred)

    metric_vars = [i for i in tf.local_variables() if 'roc_auc' in i.name.split('/')[1]]

    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


def create_model(input_shape):
    '''
    Define network
    '''
    inputs = Input(shape=input_shape)

    chexNet = DenseNet121(
          include_top=True
        , input_tensor=inputs
        , weights="ChexNet_weight.h5"
        , classes=14
    )

    chexNet = Model(
          inputs=inputs
        , outputs=chexNet.layers[-2].output
        , name="ChexNet"
    )

    model = Sequential()

    model.add(chexNet)

    model.add(Dropout(0.5, name="drop_0"))
    model.add(Dense(512, activation=None, name="dense_0"))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.5, name="drop_1"))
    model.add(Dense(1, activation="sigmoid", name="out"))

    model.summary()

    return model


def read_data(path, image_dir):
    '''
    Read and return all the image paths in directory & label of given patientId
    '''
    data = pd.read_csv(path)

    data.drop_duplicates(["patientId"], inplace=True)
    files = list(map(
        lambda x: os.path.join(image_dir, x + '.png'), data.patientId
    ))
    label = data.Target.values

    return files, label


def predict_score(model, generator, patientId):
    '''
    Predict the confidence
    '''
    score = {}

    pbar = tqdm(total=len(patientId) // BATCH_SIZE + 1, leave=False)
    for i, data in enumerate(generator):
        # predict batch of images
        pred = model.predict(data[0])
        # loop through batch
        for p, x in zip(pred, patientId[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]):
            score[x] = p[0]
        pbar.update(1)
        if len(score) >= len(patientId):
            break
    pbar.close()

    return score


def main():
    '''
    Main Method
    '''

    model = create_model(IMG_DIM)
    model.load_weights(WEIGHTS)

    gen_args = dict(
        preprocessing_function=augmentation
    )

    ###### Evaluation ######

    flow_args = dict(
          class_mode="binary"
        , batch_size=BATCH_SIZE
        , color_mode='rgb'
        , shuffle=False
        , target_size=IMG_DIM[:-1]
    )

    test_files = os.listdir(TEST_IMG_DIR)
    test_pIds = [f.split('.')[0] for f in test_files]
    test_files = [os.path.join(TEST_IMG_DIR, f) for f in test_files]

    test_gen = Generator(
          test_files
        , np.zeros([len(test_files), 1])
        , {"preprocessing_function": augmentation}
        , flow_args
    )

    test_data = {}
    for i in range(5):
        test_data['score_' + str(i)] = predict_score(model, test_gen, test_pIds)
    test_pred = pd.DataFrame(test_data)
    test_pred["score"] = test_pred[['score_' + str(i) for i in range(5)]].mean(axis=1)
    test_pred = test_pred[["score"]]

    test_pred = test_pred.reset_index().rename(columns={"index": "patientId"})

    test_pred.to_csv(os.path.join(RESULTS_DIR, "split_%d_test_classification.csv" % SPLIT_NUMBER),
                     index = False)

if __name__ == '__main__':
    main()
