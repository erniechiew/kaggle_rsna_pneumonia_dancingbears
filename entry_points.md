## Preprocessing

In `preprocessing` directory:
```
python dcm_2_png_train.py
python dcm_2_png_test.py
```

## Training
In `classification` directory
```
python classifier_train.py
```

In `detection` directory
```
python train_rsna.py
```

Note: Repeat for `SPLIT_NUMBER` 1 through 5 (modify `classifier_train.py` and `train_rsna.py` appropriately).


## Inference
In the `classification` directory
```
python classifier_predict.py
```
In `detection` directory
```
python predict_rsna.py
```
Note: Repeat for `SPLIT_NUMBER` 1 through 5 (modify `classifier_predict.py` and `predict_rsna.py` appropriately).

## Ensemble Predictions

In top level directory
```
python generate_final_prediction.py
```
