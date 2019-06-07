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

print("Finish preprocessing data and save to dataset.csv, start splitting data into training and testing")


# Preprocess phase 2 - Splitting training and testing days

dataset = pd.read_csv('dataset.csv', quoting = 3)

#Total 61 days
training_days = 55
testing_days = 6

training_dataset = dataset[(dataset["day"].values <= training_days)]
testing_dataset = dataset[(dataset["day"].values > training_days)]


print("Finish splitting data into training and testing, start plotting data into images")



# Preprocess phase 3 - Plotting data into images

training_img = []
testing_img = []
colNum = int(dataset["Ycoord"].values.max()+1)
rowNum = int(dataset["Xcoord"].values.max()+1)

# Plotting training data into images
for d in range(training_dataset["day"].values.min(),training_dataset["day"].values.max()+1):
    for t in range(int(training_dataset["normalizedTime"].values.max()+1)):
        img = np.zeros(shape=(colNum,rowNum,1))
        day = training_dataset[(training_dataset["day"].values == d)]
        daytime = day[(day["normalizedTime"].values == t)]
    
        for i in range(len(daytime)):
            X = daytime["Xcoord"].values[i]
            Y = daytime["Ycoord"].values[i]
            img[int(Y)][int(X)][0] = daytime["demand"].values[i]
        training_img.append(img)
training_img = np.array(training_img)

# Plotting testing data into images
for d in range(testing_dataset["day"].values.min(),testing_dataset["day"].values.max()+1):
    for t in range(int(testing_dataset["normalizedTime"].values.max()+1)):
        img = np.zeros(shape=(colNum,rowNum,1))
        day = testing_dataset[(testing_dataset["day"].values == d)]
        daytime = day[(day["normalizedTime"].values == t)]
    
        for i in range(len(daytime)):
            X = daytime["Xcoord"].values[i]
            Y = daytime["Ycoord"].values[i]
            img[int(Y)][int(X)][0] = daytime["demand"].values[i]
        testing_img.append(img)
testing_img = np.array(testing_img)

del d, t, i, img, day, daytime, X, Y

print("Finish plotting data into images, start preparing training data")

# Preprocess phase 4 - Preparing training data for Conv-LSTM-2D model

X_train = []
y_train = []
timestep = 48
for i in range(timestep, training_img.shape[0]):
    X_train.append(training_img[i-timestep:i,:,:,:])
    y_train.append(training_img[i-(timestep-1):i+1,:,:,:])
X_train = np.array(X_train)
y_train = np.array(y_train)

del i

print("Finish preparing training data, start preparing Conv-LSTM-2D model")


# Start preparing Conv-LSTM-2D model

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
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

seq.compile(loss='mse', optimizer='adam')

print("Finish preparing Conv-LSTM-2D model, start training!")

seq.fit(X_train, y_train, batch_size=4,
        epochs=50, validation_split=0.05)

model_name = 'conv_lstm_time48_filter32_lyr4_batch4_trainday55.h5'

seq.save(model_name) 

print("Finally finish training! Now start predicting")

#seq = load_model(model_name)

y_pred = []
y_test = testing_img[testing_img.shape[0]-5:testing_img.shape[0],:,:,0]
X_test = testing_img[testing_img.shape[0]-timestep-5:testing_img.shape[0]-5,:,:,:]
for i in range(5):
    new_pos = seq.predict(X_test[np.newaxis, ::, ::, ::, ::])
    y_pred.append(new_pos[0, -1, ::, ::, 0])
    new_pos = new_pos[0, -1, ::, ::, ::]
    X_test = np.concatenate((X_test,new_pos[np.newaxis, ::, ::, ::]))
    X_test = np.delete(X_test, (0), axis=0)
y_pred = np.array(y_pred)




print("Finish predicting, start post-processing prediction data")


newDF = pd.DataFrame(columns = ['latitude', 'longitude', 'prediction', 'demand', 'TPlus'])
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

for k in range(y_pred.shape[0]):
    TPlus = k + 1
    for i in range(y_pred.shape[1]):
        latitude = minlatitude + difflatitude * i
        for j in range(y_pred.shape[2]):
            longitude = minlongitude + difflongitude * j
            pred = pd.DataFrame({'latitude':[latitude], 'longitude': [longitude], 'prediction': [y_pred[k][i][j]], 'demand':[y_test[k][i][j]], 'TPlus':[TPlus]})
            newDF = newDF.append(pred,ignore_index=True)
        
compare = newDF[(newDF['demand'] != 0)]

fig = plt.figure()
ax = fig.add_subplot(121)
ax.set_title('actual')
plt.imshow(y_test)
ax = fig.add_subplot(122)
ax.set_title('pred')
plt.imshow(y_pred)
        
r2 = r2_score(newDF['demand'], newDF['prediction'])  
r2
        




fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_title('actual 1')
plt.imshow(y_test[0])
ax = fig.add_subplot(1,2,2)
ax.set_title('pred 1')
plt.imshow(y_pred[0])

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_title('actual 2')
plt.imshow(y_test[1])
ax = fig.add_subplot(1,2,2)
ax.set_title('pred 2')
plt.imshow(y_pred[1])

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_title('actual 3')
plt.imshow(y_test[2])
ax = fig.add_subplot(1,2,2)
ax.set_title('pred 3')
plt.imshow(y_pred[2])

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_title('actual 4')
plt.imshow(y_test[3])
ax = fig.add_subplot(1,2,2)
ax.set_title('pred 4')
plt.imshow(y_pred[3])

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.set_title('actual 5')
plt.imshow(y_test[4])
ax = fig.add_subplot(1,2,2)
ax.set_title('pred 5')
plt.imshow(y_pred[4])
