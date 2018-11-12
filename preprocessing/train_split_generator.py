import pandas as pd
import numpy as np
import os
import pydicom
import json
from sklearn.model_selection import KFold



with open("../SETTINGS.json") as f:
    SETTINGS = json.load(f)


def image_meta_extractor(img_id):

    img_dat = pydicom.read_file(os.path.join("../", SETTINGS["TRAIN_RAW_DATA_DIR"], img_id + '.dcm'))

    PatientAge = int(img_dat.PatientAge)
    PatientSex = img_dat.PatientSex
    ViewPosition = img_dat.ViewPosition

    return(pd.Series([PatientAge, PatientSex, ViewPosition],
                      index = ['PatientAge', 'PatientSex', 'ViewPosition']))

def main():

    detailed_class_info = pd.read_csv(os.path.join("../", SETTINGS["DETAILED_CLASS_INFO"]))
    train_labels = pd.read_csv(os.path.join("../", SETTINGS["TRAIN_LABELS"]))

    detailed_class_info_combined = pd.concat([train_labels,
                                             detailed_class_info.drop('patientId', axis = 1)],
                                             axis = 1)


    train_images = [f for f in os.listdir(os.path.join("../", SETTINGS["TRAIN_RAW_DATA_DIR"])) \
                      if f.endswith('.dcm')]
    train_images = [f.split('.')[0] for f in train_images]

    train_images_metadata = pd.DataFrame({'patientId' : train_images})

    train_images_metadata = pd.concat([
        train_images_metadata['patientId'],
        train_images_metadata['patientId'].apply(image_meta_extractor)],
        axis = 1)

    full_data = pd.merge(detailed_class_info_combined,
                        train_images_metadata, how = "inner",
                        on = 'patientId')

    kf = KFold(n_splits=5, random_state=586088, shuffle=False)

    patientids_ = full_data['patientId'].unique()

    i = 1

    for train_index, test_index in kf.split(patientids_):

        train_users = patientids_[train_index]
        valid_users = patientids_[test_index]

        assert(np.intersect1d(train_users, valid_users).shape[0] == 0)

        df_train = full_data[full_data['patientId'].isin(train_users)]
        df_test = full_data[full_data['patientId'].isin(valid_users)]

        df_train.to_csv(os.path.join("../", SETTINGS["TRAINING_SPLIT_DIR"],
                                            "s2_split_%d_train_data_5cv.csv" % i),
                        index = False)

        df_test.to_csv(os.path.join("../", SETTINGS["TRAINING_SPLIT_DIR"],
                                            "s2_split_%d_valid_data_5cv.csv" % i),
                        index = False)

        i += 1


if __name__ == "__main__":
    main()
