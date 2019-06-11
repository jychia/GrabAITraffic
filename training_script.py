#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import preprocessing
import zipfile
from shutil import copyfile, rmtree
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model



file_name = 'datasets/training.csv'

# Unzip training dataset and move it into datasets folder
zip_ref = zipfile.ZipFile("datasets/traffic-management.zip", 'r')
zip_ref.extractall("datasets/")
zip_ref.close()
copyfile("datasets/Traffic Management/training.csv",file_name)
rmtree("datasets/Traffic Management")

# Preprocess raw training data
edited_file_name = preprocessing.preprocessing_rawdata(file_name)

# Read the preprocessed data
dataset = pd.read_csv(edited_file_name, quoting = 3)

# Plot the data into images 
training_img = preprocessing.plot_data_to_img(dataset)

print("Start preparing training data")

X_train = []
y_train = []
timestep = 96
for i in range(timestep, training_img.shape[0]):
    X_train.append(training_img[i-timestep:i,:,:,:])
    y_train.append(training_img[i-(timestep-1):i+1,:,:,:])
X_train = np.array(X_train)
y_train = np.array(y_train)

del i

print("Finish preparing training data, start preparing model")


# Start preparing Conv-LSTM-2D model

seq = Sequential()
seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                   input_shape=(None, 46, 36, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))

seq.compile(loss='mse', optimizer='adam')

print("Finish preparing model, start training!")

seq.fit(X_train, y_train, batch_size=2,
        epochs=50, validation_split=0.05)

model_name = 'final_model.h5'

seq.save(model_name) 

print("Finally finish training!")
