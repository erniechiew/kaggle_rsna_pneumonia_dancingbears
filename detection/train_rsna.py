import os
from voc import parse_voc_annotation
from yolo import create_yolov3_model, dummy_loss
from generator import BatchGenerator
from utils.utils import normalize, evaluate, makedirs
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from callbacks import CustomModelCheckpoint, CustomTensorBoard
from utils.multi_gpu_model import multi_gpu_model
import tensorflow as tf
import keras
from keras.models import load_model

import pandas as pd
import pickle
from PIL import Image
from tqdm import tqdm
import numpy as np
import json

with open("../SETTINGS.json") as f:
    SETTINGS = json.load(f)

SPLIT_NUMBER = 4

TRAIN_FILE_PATH = os.path.join("../", SETTINGS["TRAINING_SPLIT_DIR"], "s2_split_%d_train_data_5cv.csv" % SPLIT_NUMBER)
VALID_FILE_PATH = os.path.join("../", SETTINGS["TRAINING_SPLIT_DIR"], "s2_split_%d_valid_data_5cv.csv" % SPLIT_NUMBER)
MODEL_WEIGHT_FILE = os.path.join("../", SETTINGS['DETECTION_WEIGHTS_DIR'], "yolo_fold%d.h5" % SPLIT_NUMBER)


# Obtain training and validation labels with opacities
train_data = pd.read_csv(TRAIN_FILE_PATH)
valid_data = pd.read_csv(VALID_FILE_PATH)

train_data = train_data.loc[train_data['Target'] == 1]
valid_data = valid_data.loc[valid_data['Target'] == 1]

# Convert labels to format required by YOLOv3

print("Converting annotations to appropriate format....")
train_ints = []

for img in tqdm(np.unique(train_data['patientId'])):
    tmpdict = {}
    tmpdict['filename'] = '../' + SETTINGS["TRAIN_CLEAN_DATA_DIR"] + '/' + img + '.png'

    im = Image.open('../' + SETTINGS["TRAIN_CLEAN_DATA_DIR"] + '/' + img + '.png')

    tmpdict['width'], tmpdict['height'] = im.size

    img_annotations = train_data.loc[train_data['patientId'] == img]

    object_list = []

    for row in range(img_annotations.shape[0]):
        the_row = img_annotations.iloc[row]
        smallerdict = {}
        smallerdict['name'] = 'Opacity'
        smallerdict['xmax'] = int(float(the_row['x']) + float(the_row['width']))
        smallerdict['xmin'] = int(float(the_row['x']))
        smallerdict['ymax'] = int(float(the_row['y']) + float(the_row['height']))
        smallerdict['ymin'] = int(float(the_row['y']))

        object_list.append(smallerdict)

    tmpdict['object'] = object_list

    train_ints.append(tmpdict)

valid_ints = []

for img in tqdm(np.unique(valid_data['patientId'])):
    tmpdict = {}
    tmpdict['filename'] = '../' + SETTINGS["TRAIN_CLEAN_DATA_DIR"] + '/' + img + '.png'

    im = Image.open('../' + SETTINGS["TRAIN_CLEAN_DATA_DIR"] + '/' + img + '.png')

    tmpdict['width'], tmpdict['height'] = im.size

    img_annotations = valid_data.loc[valid_data['patientId'] == img]

    object_list = []

    for row in range(img_annotations.shape[0]):
        the_row = img_annotations.iloc[row]
        smallerdict = {}
        smallerdict['name'] = 'Opacity'
        smallerdict['xmax'] = int(float(the_row['x']) + float(the_row['width']))
        smallerdict['xmin'] = int(float(the_row['x']))
        smallerdict['ymax'] = int(float(the_row['y']) + float(the_row['height']))
        smallerdict['ymin'] = int(float(the_row['y']))

        object_list.append(smallerdict)

    tmpdict['object'] = object_list

    valid_ints.append(tmpdict)

print("Done....")

def create_model(
    nb_class,
    anchors,
    max_box_per_image,
    max_grid, batch_size,
    warmup_batches,
    ignore_thresh,
    multi_gpu,
    saved_weights_name,
    lr,
    grid_scales,
    obj_scale,
    noobj_scale,
    xywh_scale,
    class_scale
):
    if multi_gpu > 1:
        with tf.device('/cpu:0'):
            template_model, infer_model = create_yolov3_model(
                nb_class            = nb_class,
                anchors             = anchors,
                max_box_per_image   = max_box_per_image,
                max_grid            = max_grid,
                batch_size          = batch_size//multi_gpu,
                warmup_batches      = warmup_batches,
                ignore_thresh       = ignore_thresh,
                grid_scales         = grid_scales,
                obj_scale           = obj_scale,
                noobj_scale         = noobj_scale,
                xywh_scale          = xywh_scale,
                class_scale         = class_scale
            )
    else:
        template_model, infer_model = create_yolov3_model(
            nb_class            = nb_class,
            anchors             = anchors,
            max_box_per_image   = max_box_per_image,
            max_grid            = max_grid,
            batch_size          = batch_size,
            warmup_batches      = warmup_batches,
            ignore_thresh       = ignore_thresh,
            grid_scales         = grid_scales,
            obj_scale           = obj_scale,
            noobj_scale         = noobj_scale,
            xywh_scale          = xywh_scale,
            class_scale         = class_scale
        )

    # load the pretrained weight if exists, otherwise load the backend weight only
    if os.path.exists(saved_weights_name):
        print("\nLoading pretrained weights.\n")
        template_model.load_weights(saved_weights_name)
    else:
        template_model.load_weights("backend.h5", by_name=True)

    if multi_gpu > 1:
        train_model = multi_gpu_model(template_model, gpus=multi_gpu)
    else:
        train_model = template_model

    optimizer = Adam(lr=lr, clipnorm=0.001)
    train_model.compile(loss=dummy_loss, optimizer=optimizer)

    return train_model, infer_model


def create_callbacks(saved_weights_name, model_to_save):

    early_stop = EarlyStopping(
        monitor     = 'val_loss',
        min_delta   = 0.01,
        patience    = 8,
        mode        = 'min',
        verbose     = 1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save   = model_to_save,
        filepath        = saved_weights_name,# + '{epoch:02d}.h5',
        monitor         = 'val_loss',
        verbose         = 1,
        save_best_only  = True,
        mode            = 'min',
        period          = 1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.1,
        patience = 3,
        verbose  = 1,
        mode     = 'min',
        epsilon  = 0.01,
        cooldown = 0,
        min_lr   = 0
    )

    return [early_stop, checkpoint, reduce_on_plateau]


# PARAMS
ANCHORS = SETTINGS['ANCHOR_SPLIT_%d' % SPLIT_NUMBER]


TRAIN_TIMES = 1
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NB_EPOCHS = 50
WARMUP_EPOCHS = 3
IGNORE_THRES = 0.5

GRID_SCALES = [1, 1, 1]
OBJ_SCALE = 5
NOOBJ_SCALE = 1
XYWH_SCALE = 1
CLASS_SCALE = 1

MAX_INPUT_SIZE = 608
MIN_INPUT_SIZE = 608

DEBUG = True

max_box_per_image = max([len(inst['object']) for inst in (train_ints + valid_ints)])

labels = ['Opacity']


train_generator = BatchGenerator(
    instances           = train_ints,
    anchors             = ANCHORS,
    labels              = labels,
    downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
    max_box_per_image   = max_box_per_image,
    batch_size          = BATCH_SIZE,
    min_net_size        = MIN_INPUT_SIZE,
    max_net_size        = MAX_INPUT_SIZE,
    shuffle             = True,
    jitter              = 0.3,
    norm                = normalize
)

valid_generator = BatchGenerator(
    instances           = valid_ints,
    anchors             = ANCHORS,
    labels              = labels,
    downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
    max_box_per_image   = max_box_per_image,
    batch_size          = BATCH_SIZE,
    min_net_size        = MIN_INPUT_SIZE,
    max_net_size        = MAX_INPUT_SIZE,
    shuffle             = True,
    jitter              = 0.0,
    norm                = normalize
)

WARMUP_BATCHES = WARMUP_EPOCHS * (TRAIN_TIMES * len(train_generator))


train_model, infer_model = create_model(
    nb_class            = len(labels),
    anchors             = ANCHORS,
    max_box_per_image   = max_box_per_image,
    max_grid            = [MAX_INPUT_SIZE, MAX_INPUT_SIZE],
    batch_size          = BATCH_SIZE,
    warmup_batches      = WARMUP_BATCHES,
    ignore_thresh       = IGNORE_THRES,
    multi_gpu           = 0,
    saved_weights_name  = MODEL_WEIGHT_FILE,
    lr                  = LEARNING_RATE,
    grid_scales         = GRID_SCALES,
    obj_scale           = OBJ_SCALE,
    noobj_scale         = NOOBJ_SCALE,
    class_scale         = CLASS_SCALE,
    xywh_scale          = XYWH_SCALE,
)


train_model.summary()


callbacks = create_callbacks(MODEL_WEIGHT_FILE, infer_model)


train_model.fit_generator(
    generator        = train_generator,
    steps_per_epoch  = len(train_generator) * TRAIN_TIMES,
    epochs           = NB_EPOCHS,
    verbose          = 1,
    callbacks        = callbacks,
    validation_data  = valid_generator,
    validation_steps = len(valid_generator),
    max_queue_size   = 8
)


print("Done training")
