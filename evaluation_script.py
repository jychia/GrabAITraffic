#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import geohash
import numpy as np
import os
import pandas as pd
import preprocessing
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model



# Allow user to enter file path for evaluation dataset
user_input = input("Enter the file path of your evaluation dataset: ")
assert os.path.exists(user_input), "Invalid file at " + str(user_input)
f = open(user_input,'r+')
print("Valid file path entered")
f.close()


# Preprocess the raw data from evaluation dataset
edited_file_name = preprocessing.preprocessing_rawdata(user_input)

dataset = pd.read_csv(edited_file_name, quoting = 3)

# Transform the processed data into 2D data plots
testing_img = preprocessing.plot_data_to_img(dataset)

# Load neural network model
model_name = 'models/final_model.h5'
seq = load_model(model_name)

# Run predictions for T+1 to T+5

# First method of prediction: 
# - Feed the last 96 demand plots into the model (T-96 to T)
# - Model prediction is T-95 to T+1
# - Extract the last frame from prediction results T+1
# - Add it into X_test, then remove the first frame from X_test
# - Now X_test is T-95 to T+1 where T-95 till T are real data and T+1 is previous prediction
# - Feed X_test back to model and predict T+2
# - Repeat steps till T+5 is predicted
# This prediction method produce a highly accurate T+1 prediction, 
# but accuracy drops rapidly across predictions until T+5,
# because a prediction is fed back to model to produce further predictions,
y_pred = []
timestep = 96
X_test = testing_img[testing_img.shape[0]-timestep:testing_img.shape[0],:,:,:]
for i in range(5):
    new_pos = seq.predict(X_test[np.newaxis, ::, ::, ::, ::])
    y_pred.append(new_pos[0, -1, ::, ::, 0])
    new_pos = new_pos[0, -1, ::, ::, ::]
    X_test = np.concatenate((X_test,new_pos[np.newaxis, ::, ::, ::]))
    X_test = np.delete(X_test, (0), axis=0)
y_pred = np.array(y_pred)

# Second method of prediction:
# - Feed the last 91 demand plots into the model (T-91 to T)
# - Model prediction is T-91 to T+5
# - Extract the last 5 frames as prediction results (T+1 to T+5)
# This prediction method produces less accurate predictions,
# but accuracy of prediction across T+1 to T+5 does not vary,
# as this is because less data is passed into model to predict but data is constant across predictions
X_test2 = testing_img[testing_img.shape[0]-timestep+5:testing_img.shape[0],:,:,:]
new_pos2 = seq.predict(X_test2[np.newaxis, ::, ::, ::, ::])
y_pred2 = new_pos2[0, new_pos2.shape[1]-5:new_pos2.shape[1], ::, ::, 0]

# Extract T+1 till T+3 from first method, T+4 till T+5 from second method
y_pred_final = np.concatenate((y_pred[:3],y_pred2[3:]))

print("Finish predicting, start post-processing prediction data")

# Post process the predictions into data
final_pred = pd.DataFrame(columns = ['latitude', 'longitude', 'geohash6', 'prediction', 'TPlus'])
uniquelatitude = dataset['latitude'].unique().tolist()
uniquelatitude.sort()
difflatitude = uniquelatitude[1] - uniquelatitude[0]
minlatitude = uniquelatitude[0]

uniquelongitude = dataset['longitude'].unique().tolist()
uniquelongitude.sort()
difflongitude = uniquelongitude[1] - uniquelongitude[0]
minlongitude = uniquelongitude[0]

for k in range(y_pred_final.shape[0]):
    for i in range(y_pred_final.shape[1]):
        latitude = minlatitude + difflatitude * i
        for j in range(y_pred_final.shape[2]):
            longitude = minlongitude + difflongitude * j
            pred = pd.DataFrame({'latitude':[latitude], \
                                 'longitude':[longitude], \
                                 'geohash6':[geohash.encode(latitude,longitude,6)], \
                                 'prediction':[y_pred_final[k][i][j]], \
                                 'TPlus':[k+1]})
            final_pred = final_pred.append(pred,ignore_index=True)
            
final_pred.to_csv("predictions.csv", index  = False)
        
del uniquelatitude, difflatitude, minlatitude
del uniquelongitude, difflongitude, minlongitude
del i, j, k, pred

