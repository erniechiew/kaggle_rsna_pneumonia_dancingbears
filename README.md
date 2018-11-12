# RSNA Pneumonia Detection Challenge: 11th Place Solution Code

## Hardware Used
* Intel Core i5-7600k, 4 Cores 4 Threads
* 32 GB RAM
* 1 x NVIDIA GeForce GTX 1080 TI

## Platform and Software
* Windows 10 (Version 1803)
* Python 3.5.4
* CUDA 9.0
* cuDNN 7.0

## Important Note

The training and inference steps will overwrite any existing weights and test set predictions.


## Data

Data should be obtained from the RSNA Pneumonia Detection Challenge page on Kaggle and placed in the `data` directory. For recreating our solution using the Stage 2 data, the `data` directory should contain:

1. The folder `stage_2_train_images` containing the Stage 2 training images.
2. The folder `stage_2_test_images` containing the Stage 2 test images.
3. The folder `stage_2_train_images_png` containing the Stage 2 training images in PNG format (generated in preprocessing step).
4. The folder `stage_2_train_images_png` containing the Stage 2 training images in PNG format (generated in preprocessing step).
5. `stage_2_detailed_class_info.csv`
6. `stage_2_train_labels.csv`
7. `stage_2_sample_submission.csv`


## Preprocessing

### Convert DCM to PNG

In the `preprocessing` directory, run
```
python dcm_2_png_train.py
python dcm_2_png_test.py
```
which will convert the training and test images from DCM to PNG format, respectively. The PNG images will be stored in `data/stage_2_train_images_png` and `data/stage_2_test_images_png`.

### Generate 5-fold CV splits (Optional)

We provide the splits we used for our final submission in `training_splits/`. But if you wish to generate new splits (and overwrite the provided splits), in the `preprocessing` directory, run
```
python train_split_generator.py
```
Generated splits will be stored in `training_splits/` folder.


## Training

### Classification Model

In the `classification` directory, run
```
python classifier_train.py
```
Resulting model weights will be stored in `classification/weights/` folder. Modify `SPLIT_NUMBER` (manually within the `classifier_train.py` file for now -- sorry) to train the model for each of the 5 splits. After this step, the `classification/weights/` folder should contain the trained weights from each of the five folds.


### Bounding-Box Detection Model

In the `detection` directory, run
```
python train_rsna.py
```
Resulting model weights will be stored in `detection/weights/` folder. Modify `SPLIT_NUMBER` (manually within the `train_rsna.py` file for now -- sorry) to train the model for each of the 5 splits. After this step, the `detection/weights/` folder should contain the trained weights from each of the five folds.

(Optional) You may opt to generate your own anchors. This can be done by running
```
python gen_anchors.py --split N
```
where `N` is the desired split (1 to 5). Replace the output list accordingly in `SETTINGS.txt`.


## Inference

### Classification Model

In the `classification` directory, run
```
python classifier_predict.py
```
Resulting predictions on test set will be stored in `classification/results/` folder. Again, modify `SPLIT_NUMBER` in `classifier_predict.py` to train the model for each of the 5 splits. After this step, the `classification/results/` folder should contain the classifier's predictions on the test set for each of the 5 folds.

### Bounding-Box Detection Model

In the `detection` directory, run
```
python predict_rsna.py
```
Resulting predictions on test set will be stored in `detection/results/` folder. Again, modify `SPLIT_NUMBER` in `predict_rsna.py` to train the model for each of the 5 splits. After this step, the `detection/results/` folder should contain predictions on the test set for each of the 5 folds.


## Ensemble Predictions

To generate final ensembled predictions on Stage 2 test data, run
```
python generate_final_prediction.py
```



## Acknowledgements

We used code and pre-trained weights from the following repositories:

https://github.com/brucechou1983/CheXNet-Keras

https://github.com/experiencor/keras-yolo3


