# -*- coding: utf-8 -*-
from PIL import Image
from tqdm import tqdm

import multiprocessing
import os
import pydicom

import json


with open("../SETTINGS.json") as f:
    SETTINGS = json.load(f)

SRC_PATH = os.path.join("../", SETTINGS["TRAIN_RAW_DATA_DIR"])
DST_PATH  = os.path.join("../", SETTINGS["TRAIN_CLEAN_DATA_DIR"])

def convert(patientId, src_path=SRC_PATH, dst_path=DST_PATH):
    '''
    Convert dcm to png
    '''
    Image.fromarray(
        pydicom.dcmread(
            os.path.join(SRC_PATH, patientId + '.dcm')
        ).pixel_array
    ).save(os.path.join(DST_PATH, patientId + '.png'))


def main():
    '''
    Main Method
    '''
    assert os.path.lexists(SRC_PATH)
    assert os.path.lexists(DST_PATH)

    patient_ids = [os.path.splitext(f)[0] for f in os.listdir(SRC_PATH)]

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        pbar = tqdm(
              total=len(patient_ids)
            , leave=False
            , desc="Converting"
        )
        for _ in p.imap_unordered(convert, patient_ids):
            pbar.update(1)
        pbar.close()

    print("[Done]!")


if __name__ == '__main__':
    main()
