import geohash
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.metrics import r2_score


# Preprocess phase 1 - Process raw data & calculate extra columns

print("Start reading raw data")

dataset = pd.read_csv('training.csv', quoting = 3)

print("Finish reading raw data, start preprocessing data phase 1")

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
numlongitude = len(uniquelongitude)
difflongitude = uniquelongitude[1] - uniquelongitude[0]
minlongitude = uniquelongitude[0]

Xcoord = (dataset['longitude'].values - minlongitude) / difflongitude
Ycoord = (dataset['latitude'].values - minlatitude) / difflatitude

dataset["Xcoord"] = Xcoord
dataset["Ycoord"] = Ycoord

dayofweek = dataset['day'].values % 7
dataset["dayOfWeek"] = dayofweek

del Xcoord, Ycoord, dayofweek
del col, difflatitude, difflongitude, i, minlatitude, minlongitude, row
del uniquelongitude, uniquelatitude, numlatitude, numlongitude

dataset.to_csv(r'dataset.csv', index  = False)

print("Finish preprocessing data and save to dataset.csv, start plotting data into images")




# Preprocess phase 2 - Plotting data into images

dataset = pd.read_csv('dataset.csv', quoting = 3)
imgseries = []

for d in range(1,dataset["day"].values.max()+1):
    for t in range(int(dataset["normalizedTime"].values.max()+1)):
        img = np.zeros(shape=(int(dataset["Ycoord"].values.max()+1),int(dataset["Xcoord"].values.max()+1),1))
        day = dataset[(dataset["day"].values == d)]
        daytime = day[(day["normalizedTime"].values == t)]
    
        for i in range(len(daytime)):
            X = daytime["Xcoord"].values[i]
            Y = daytime["Ycoord"].values[i]
            img[int(Y)][int(X)][0] = daytime["demand"].values[i]
        imgseries.append(img)
imgseries = np.array(imgseries)

del d, t, i, img, day, daytime, X, Y

print("Finish plotting data into images, start preparing training and testing data")

# Preprocess phase 3 - Preparing training and testing data for Conv-LSTM-2D model

X_train = []
y_train = []
length = imgseries.shape[0]-5
timestep = 48
for i in range(timestep, length):
    X_train.append(imgseries[i-timestep:i,:,:,:])
    y_train.append(imgseries[i-(timestep-1):i+1,:,:,:])
X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = imgseries[length-timestep:length,:,:,:]
y_test = imgseries[length,:,:,0]

del i

print("Finish preparing training and testing data, start preparing Conv-LSTM-2D model")


# Start preparing Conv-LSTM-2D model

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.utils import multi_gpu_model
from keras.models import load_model

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
#seq = multi_gpu_model(seq)
seq.compile(loss='mse', optimizer='adam')

print("Finish preparing Conv-LSTM-2D model, start training!")

seq.fit(X_train, y_train, batch_size=4,
        epochs=50, validation_split=0.05)

model_name = 'conv_lstm_time48_filter32_lyr4_batch4.h5'

seq.save(model_name) 

print("Finally finish training! Now start predicting")

#seq = load_model(model_name)




new_pos = seq.predict(X_test[np.newaxis, ::, ::, ::, ::])
y_pred = new_pos[0, -1, ::, ::, 0]


print("Finish predicting, start post-processing prediction data")


newDF = pd.DataFrame(columns = ['latitude', 'longitude', 'demand_prediction', 'demand'])
uniquelatitude = dataset['latitude'].unique().tolist()
uniquelatitude.sort()
numlatitude = len(uniquelatitude)
difflatitude = uniquelatitude[1] - uniquelatitude[0]
minlatitude = uniquelatitude[0]

uniquelongitude = dataset['longitude'].unique().tolist()
uniquelongitude.sort()
numlongitude = len(uniquelongitude)
difflongitude = uniquelongitude[1] - uniquelongitude[0]
minlongitude = uniquelongitude[0]


for i in range(y_pred.shape[0]):
    latitude = minlatitude + difflatitude * i
    for j in range(y_pred.shape[1]):
        longitude = minlongitude + difflongitude * j
        pred = pd.DataFrame({'latitude':[latitude], 'longitude': [longitude], 'demand_prediction': [y_pred[i][j]], 'demand':[y_test[i][j]]})
        newDF = newDF.append(pred,ignore_index=True)
        
compare = newDF[(newDF['demand'] != 0)]
normalizedDT = dataset[(dataset["normalizedDayTime"].values == length)]

fig = plt.figure()
ax = fig.add_subplot(121)
ax.set_title('actual')
plt.imshow(y_test)
ax = fig.add_subplot(122)
ax.set_title('pred')
plt.imshow(y_pred)
        
r2 = r2_score(compare['demand'], compare['demand_prediction'])  
r2
        