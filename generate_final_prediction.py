import pandas as pd
import numpy as np
import os
import json

from utils import *

with open("SETTINGS.json") as f:
    SETTINGS = json.load(f)

print("Loading and processing YOLO predictions...")

# Merge yolo predictions:

yolo_1 = pd.read_csv(os.path.join(SETTINGS['DETECTION_RESULTS_DIR'], "yolo_fold1_predictions_on_test.csv"))
yolo_1_flip = pd.read_csv(os.path.join(SETTINGS['DETECTION_RESULTS_DIR'], "yolo_fold1_predictions_on_test_flipped.csv"))

yolo_2 = pd.read_csv(os.path.join(SETTINGS['DETECTION_RESULTS_DIR'], "yolo_fold2_predictions_on_test.csv"))
yolo_2_flip = pd.read_csv(os.path.join(SETTINGS['DETECTION_RESULTS_DIR'], "yolo_fold2_predictions_on_test_flipped.csv"))

yolo_3 = pd.read_csv(os.path.join(SETTINGS['DETECTION_RESULTS_DIR'], "yolo_fold3_predictions_on_test.csv"))
yolo_3_flip = pd.read_csv(os.path.join(SETTINGS['DETECTION_RESULTS_DIR'], "yolo_fold3_predictions_on_test_flipped.csv"))

yolo_4 = pd.read_csv(os.path.join(SETTINGS['DETECTION_RESULTS_DIR'], "yolo_fold4_predictions_on_test.csv"))
yolo_4_flip = pd.read_csv(os.path.join(SETTINGS['DETECTION_RESULTS_DIR'], "yolo_fold4_predictions_on_test_flipped.csv"))

yolo_5 = pd.read_csv(os.path.join(SETTINGS['DETECTION_RESULTS_DIR'], "yolo_fold5_predictions_on_test.csv"))
yolo_5_flip =pd.read_csv(os.path.join(SETTINGS['DETECTION_RESULTS_DIR'], "yolo_fold5_predictions_on_test_flipped.csv"))


yolo_1['PredictionString'] = yolo_1['PredictionString'].apply(lambda x: cutoffer(x, 0.8))
yolo_1['PredictionString'] = yolo_1['PredictionString'].apply(lambda x: truncate_predstring_based_on_nms(x, nms(predstring_to_list(x), 0.1)))
yolo_1_flip['PredictionString'] = yolo_1_flip['PredictionString'].apply(lambda x: cutoffer(x, 0.8))
yolo_1_flip['PredictionString'] = yolo_1_flip['PredictionString'].apply(lambda x: truncate_predstring_based_on_nms(x, nms(predstring_to_list(x), 0.1)))


yolo_2['PredictionString'] = yolo_2['PredictionString'].apply(lambda x: cutoffer(x, 0.8))
yolo_2['PredictionString'] = yolo_2['PredictionString'].apply(lambda x: truncate_predstring_based_on_nms(x, nms(predstring_to_list(x), 0.1)))
yolo_2_flip['PredictionString'] = yolo_2_flip['PredictionString'].apply(lambda x: cutoffer(x, 0.8))
yolo_2_flip['PredictionString'] = yolo_2_flip['PredictionString'].apply(lambda x: truncate_predstring_based_on_nms(x, nms(predstring_to_list(x), 0.1)))


yolo_3['PredictionString'] = yolo_3['PredictionString'].apply(lambda x: cutoffer(x, 0.8))
yolo_3['PredictionString'] = yolo_3['PredictionString'].apply(lambda x: truncate_predstring_based_on_nms(x, nms(predstring_to_list(x), 0.1)))
yolo_3_flip['PredictionString'] = yolo_3_flip['PredictionString'].apply(lambda x: cutoffer(x, 0.8))
yolo_3_flip['PredictionString'] = yolo_3_flip['PredictionString'].apply(lambda x: truncate_predstring_based_on_nms(x, nms(predstring_to_list(x), 0.1)))


yolo_4['PredictionString'] = yolo_4['PredictionString'].apply(lambda x: cutoffer(x, 0.8))
yolo_4['PredictionString'] = yolo_4['PredictionString'].apply(lambda x: truncate_predstring_based_on_nms(x, nms(predstring_to_list(x), 0.1)))
yolo_4_flip['PredictionString'] = yolo_4_flip['PredictionString'].apply(lambda x: cutoffer(x, 0.8))
yolo_4_flip['PredictionString'] = yolo_4_flip['PredictionString'].apply(lambda x: truncate_predstring_based_on_nms(x, nms(predstring_to_list(x), 0.1)))


yolo_5['PredictionString'] = yolo_5['PredictionString'].apply(lambda x: cutoffer(x, 0.8))
yolo_5['PredictionString'] = yolo_5['PredictionString'].apply(lambda x: truncate_predstring_based_on_nms(x, nms(predstring_to_list(x), 0.1)))
yolo_5_flip['PredictionString'] = yolo_5_flip['PredictionString'].apply(lambda x: cutoffer(x, 0.8))
yolo_5_flip['PredictionString'] = yolo_5_flip['PredictionString'].apply(lambda x: truncate_predstring_based_on_nms(x, nms(predstring_to_list(x), 0.1)))


# Reflip flipped boxes
yolo_1_flip['PredictionString'] = yolo_1_flip["PredictionString"].apply(bounding_box_lr_flip)
yolo_2_flip['PredictionString'] = yolo_2_flip["PredictionString"].apply(bounding_box_lr_flip)
yolo_3_flip['PredictionString'] = yolo_3_flip["PredictionString"].apply(bounding_box_lr_flip)
yolo_4_flip['PredictionString'] = yolo_4_flip["PredictionString"].apply(bounding_box_lr_flip)
yolo_5_flip['PredictionString'] = yolo_5_flip["PredictionString"].apply(bounding_box_lr_flip)


# Merge predictions on flipped and unflipped
yolo_1 = yolo_1.merge(yolo_1_flip, on = "patientId")
yolo_2 = yolo_2.merge(yolo_2_flip, on = "patientId")
yolo_3 = yolo_3.merge(yolo_3_flip, on = "patientId")
yolo_4 = yolo_4.merge(yolo_4_flip, on = "patientId")
yolo_5 = yolo_5.merge(yolo_5_flip, on = "patientId")

yolo_1.loc[pd.isnull(yolo_1['PredictionString_x']), 'PredictionString_x'] = ""
yolo_1.loc[pd.isnull(yolo_1['PredictionString_y']), 'PredictionString_y'] = ""

yolo_2.loc[pd.isnull(yolo_2['PredictionString_x']), 'PredictionString_x'] = ""
yolo_2.loc[pd.isnull(yolo_2['PredictionString_y']), 'PredictionString_y'] = ""

yolo_3.loc[pd.isnull(yolo_3['PredictionString_x']), 'PredictionString_x'] = ""
yolo_3.loc[pd.isnull(yolo_3['PredictionString_y']), 'PredictionString_y'] = ""

yolo_4.loc[pd.isnull(yolo_4['PredictionString_x']), 'PredictionString_x'] = ""
yolo_4.loc[pd.isnull(yolo_4['PredictionString_y']), 'PredictionString_y'] = ""

yolo_5.loc[pd.isnull(yolo_5['PredictionString_x']), 'PredictionString_x'] = ""
yolo_5.loc[pd.isnull(yolo_5['PredictionString_y']), 'PredictionString_y'] = ""

yolo_1['PredictionString'] = yolo_1['PredictionString_x'] + " " + yolo_1['PredictionString_y']
yolo_2['PredictionString'] = yolo_2['PredictionString_x'] + " " + yolo_2['PredictionString_y']
yolo_3['PredictionString'] = yolo_3['PredictionString_x'] + " " + yolo_3['PredictionString_y']
yolo_4['PredictionString'] = yolo_4['PredictionString_x'] + " " + yolo_4['PredictionString_y']
yolo_5['PredictionString'] = yolo_5['PredictionString_x'] + " " + yolo_5['PredictionString_y']

yolo_1 = yolo_1[['patientId', 'PredictionString']]
yolo_2 = yolo_2[['patientId', 'PredictionString']]
yolo_3 = yolo_3[['patientId', 'PredictionString']]
yolo_4 = yolo_4[['patientId', 'PredictionString']]
yolo_5 = yolo_5[['patientId', 'PredictionString']]



yolo_1['PredictionString'] = yolo_1['PredictionString'].apply(lambda x: get_predstring_from_box_list(combine_boxes_intersect(get_box_list_from_predstring(x))))
yolo_2['PredictionString'] = yolo_2['PredictionString'].apply(lambda x: get_predstring_from_box_list(combine_boxes_intersect(get_box_list_from_predstring(x))))
yolo_3['PredictionString'] = yolo_3['PredictionString'].apply(lambda x: get_predstring_from_box_list(combine_boxes_intersect(get_box_list_from_predstring(x))))
yolo_4['PredictionString'] = yolo_4['PredictionString'].apply(lambda x: get_predstring_from_box_list(combine_boxes_intersect(get_box_list_from_predstring(x))))
yolo_5['PredictionString'] = yolo_5['PredictionString'].apply(lambda x: get_predstring_from_box_list(combine_boxes_intersect(get_box_list_from_predstring(x))))

yolo_1.loc[pd.isnull(yolo_1['PredictionString']), 'PredictionString'] = ""
yolo_2.loc[pd.isnull(yolo_2['PredictionString']), 'PredictionString'] = ""
yolo_3.loc[pd.isnull(yolo_3['PredictionString']), 'PredictionString'] = ""
yolo_4.loc[pd.isnull(yolo_4['PredictionString']), 'PredictionString'] = ""
yolo_5.loc[pd.isnull(yolo_5['PredictionString']), 'PredictionString'] = ""
yolo_1 = yolo_1.merge(yolo_2, on = "patientId")
yolo_1["PredictionString"] = yolo_1['PredictionString_x'] + " " + yolo_1["PredictionString_y"]
yolo_1["PredictionString"] = yolo_1['PredictionString'].apply(lambda x: x.strip())

yolo_1 = yolo_1[['patientId', 'PredictionString']]

yolo_1 = yolo_1.merge(yolo_3, on = "patientId")
yolo_1["PredictionString"] = yolo_1['PredictionString_x'] + " " + yolo_1["PredictionString_y"]
yolo_1["PredictionString"] = yolo_1['PredictionString'].apply(lambda x: x.strip())

yolo_1 = yolo_1[['patientId', 'PredictionString']]

yolo_1 = yolo_1.merge(yolo_4, on = "patientId")
yolo_1["PredictionString"] = yolo_1['PredictionString_x'] + " " + yolo_1["PredictionString_y"]
yolo_1["PredictionString"] = yolo_1['PredictionString'].apply(lambda x: x.strip())

yolo_1 = yolo_1[['patientId', 'PredictionString']]

yolo_1 = yolo_1.merge(yolo_5, on = "patientId")
yolo_1["PredictionString"] = yolo_1['PredictionString_x'] + " " + yolo_1["PredictionString_y"]
yolo_1["PredictionString"] = yolo_1['PredictionString'].apply(lambda x: x.strip())

yolo_1 = yolo_1[['patientId', 'PredictionString']]


yolo_1['PredictionString'] = yolo_1['PredictionString'].apply(sort_predstring_by_confidence)

yolo_1['PredictionString'] = yolo_1['PredictionString'].apply(lambda x: get_predstring_from_box_list(combine_boxes_intersect(count_boxes_intersect(get_box_list_from_predstring(x)))))
yolo_1.loc[pd.isnull(yolo_1['PredictionString']), 'PredictionString'] = ""

yolo_1 = yolo_1[['patientId', 'PredictionString']]



yolo_1['PredictionString'] = yolo_1['PredictionString'].apply(lambda x: get_predstring_from_box_list(combine_boxes_intersect(get_box_list_from_predstring(x))))


print("Loading and processing classification results...")

classification_test_results1 = pd.read_csv(os.path.join(SETTINGS['CLASSIFICATION_RESULTS_DIR'], "split_1_test_classification.csv"))
classification_test_results2 = pd.read_csv(os.path.join(SETTINGS['CLASSIFICATION_RESULTS_DIR'], "split_2_test_classification.csv"))
classification_test_results3 = pd.read_csv(os.path.join(SETTINGS['CLASSIFICATION_RESULTS_DIR'], "split_3_test_classification.csv"))
classification_test_results4 = pd.read_csv(os.path.join(SETTINGS['CLASSIFICATION_RESULTS_DIR'], "split_4_test_classification.csv"))
classification_test_results5 = pd.read_csv(os.path.join(SETTINGS['CLASSIFICATION_RESULTS_DIR'], "split_5_test_classification.csv"))

classification_test_results1 = classification_test_results1.merge(classification_test_results2, on = "patientId")
classification_test_results1 = classification_test_results1.merge(classification_test_results3, on = "patientId")
classification_test_results1 = classification_test_results1.merge(classification_test_results4, on = "patientId")
classification_test_results1 = classification_test_results1.merge(classification_test_results5, on = "patientId")

classification_test_results1['avgclassifierScore'] = classification_test_results1.iloc[:, 1:6].mean(axis = 1)


print("Combining YOLO and classification results...")

yolo_1 = yolo_1.merge(classification_test_results1[['patientId', 'avgclassifierScore']], on = "patientId")
yolo_1.loc[yolo_1['avgclassifierScore'] < 0.2, 'PredictionString'] = np.nan
yolo_1 = yolo_1[['patientId', 'PredictionString']]

yolo_1.to_csv("final_submission.csv", index = False)

print("Done")
