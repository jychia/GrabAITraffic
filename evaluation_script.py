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





user_input = input("Enter the file path of your evaluation dataset: ")

assert os.path.exists(user_input), "Invalid file at " + str(user_input)
f = open(user_input,'r+')
print("Valid file path entered")
f.close()

edited_file_name = preprocessing.preprocessing_rawdata(user_input)

dataset = pd.read_csv(edited_file_name, quoting = 3)

testing_img = preprocessing.plot_data_to_img(dataset)

model_name = 'final_model.h5'
seq = load_model(model_name)

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




print("Finish predicting, start post-processing prediction data")


final_pred = pd.DataFrame(columns = ['latitude', 'longitude', 'geohash6', 'prediction', 'TPlus'])
uniquelatitude = dataset['latitude'].unique().tolist()
uniquelatitude.sort()
difflatitude = uniquelatitude[1] - uniquelatitude[0]
minlatitude = uniquelatitude[0]

uniquelongitude = dataset['longitude'].unique().tolist()
uniquelongitude.sort()
difflongitude = uniquelongitude[1] - uniquelongitude[0]
minlongitude = uniquelongitude[0]

for k in range(y_pred.shape[0]):
    for i in range(y_pred.shape[1]):
        latitude = minlatitude + difflatitude * i
        for j in range(y_pred.shape[2]):
            longitude = minlongitude + difflongitude * j
            pred = pd.DataFrame({'latitude':[latitude], \
                                 'longitude':[longitude], \
                                 'geohash6':[geohash.encode(latitude,longitude,6)], \
                                 'prediction':[y_pred[k][i][j]], \
                                 'TPlus':[k+1]})
            final_pred = final_pred.append(pred,ignore_index=True)
            
final_pred.to_csv("predictions.csv", index  = False)
        
del uniquelatitude, difflatitude, minlatitude
del uniquelongitude, difflongitude, minlongitude
del i, j, k, pred

