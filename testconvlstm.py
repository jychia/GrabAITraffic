import geohash
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

dataset = pd.read_csv('training.csv', quoting = 3)

longitude = []
latitude = []

for g in dataset["geohash6"].values:
    la, lo = geohash.decode(g)
    latitude.append(la)
    longitude.append(lo)
    
dataset["latitude"] = latitude
dataset["longitude"] = longitude

del latitude, longitude, g, la, lo

row, col = dataset.shape

normalizedTime = []
normalizedDayTime = []

for i in range(row):
    time = dataset["timestamp"][i].split(":")
    a = (int(time[0]) * 4) + (int(time[1]) / 15)
    normalizedTime.append(a)
    b = (dataset["day"][i]-1) * 96 + a
    normalizedDayTime.append(b)

dataset["normalizedTime"] = normalizedTime
dataset["normalizedDayTime"] = normalizedDayTime

del normalizedTime, normalizedDayTime, time, a, b

uniquelatitude = dataset['latitude'].unique().tolist()
uniquelatitude.sort()
numlatitude = len(uniquelatitude)
difflatitude = uniquelatitude[1] - uniquelatitude[0]
minlatitude = uniquelatitude[0]

uniquelongitude = dataset['longitude'].unique().tolist()
uniquelongitude.sort()
numlongtitude = len(uniquelongitude)
difflongitude = uniquelongitude[1] - uniquelongitude[0]
minlongtitude = uniquelongitude[0]

Xcoord = (dataset['longitude'].values - minlongtitude) / difflongitude
Ycoord = (dataset['latitude'].values - minlatitude) / difflatitude

dataset["Xcoord"] = Xcoord
dataset["Ycoord"] = Ycoord

dayofweek = dataset['day'].values % 7
dataset["dayOfWeek"] = dayofweek

del Xcoord, Ycoord, dayofweek
del col, difflatitude, difflongitude, i, minlatitude, minlongtitude, row
del uniquelongitude, uniquelatitude

dataset.to_csv(r'dataset.csv', index  = False)


dataset = pd.read_csv('dataset.csv', quoting = 3)
imgseries = []

for d in range(1,dataset["day"].values.max()+1):
    for t in range(int(dataset["normalizedTime"].values.max()+1)):
        img = np.zeros(shape=(numlatitude,numlongtitude))
        day = dataset[(dataset["day"].values == d)]
        daytime = day[(day["normalizedTime"].values == t)]
    
        for i in range(len(daytime)):
            X = daytime["Xcoord"].values[i]
            Y = daytime["Ycoord"].values[i]
            img[int(Y)][int(X)] = daytime["demand"].values[i]
        imgseries.append(img)
        
        
        
        
        
X_train = []
y_train = []
for i in range(96, int(len(imgseries)-5)):
    X_train.append(imgseries[i-96:i])
    y_train.append(imgseries[i])

for i in range(len(y_train)):
    y_train[i] = y_train[i].flatten()

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1))

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import InputLayer
from keras.layers.convolutional_recurrent import ConvLSTM2D


model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(3,3),
                   input_shape=(96, 46, 36, 1), 
                   padding='same', return_sequences=True,
                   activation='relu'))
model.add(Dropout(0.2))
model.add(ConvLSTM2D(filters=64, kernel_size=(2,2),
                   padding='same', return_sequences=True,
                   activation='relu'))
model.add(Dropout(0.2))
model.add(ConvLSTM2D(filters=64, kernel_size=(2,2),
                   padding='same', return_sequences=True,
                   activation='relu'))
model.add(Dropout(0.2))
model.add(ConvLSTM2D(filters=64, kernel_size=(2,2),
                   padding='same', 
                   activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(units = 1656))
print (model.output_shape)

model.compile(optimizer = 'adam', loss = 'mse')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)



































