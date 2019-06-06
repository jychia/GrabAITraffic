import geohash
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor

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
#for i in range(len(uniquelatitude)-1):
#    diff = uniquelatitude[i+1] - uniquelatitude[i]
#    print(diff)

uniquelongitude = dataset['longitude'].unique().tolist()
uniquelongitude.sort()
numlongitude = len(uniquelongitude)
difflongitude = uniquelongitude[1] - uniquelongitude[0]
minlongitude = uniquelongitude[0]
#for i in range(len(uniquelongitude)-1):
#    diff = uniquelongitude[i+1] - uniquelongitude[i]
#    print(diff)

Xcoord = int((dataset['longitude'].values - minlongitude) / difflongitude)
Ycoord = int((dataset['latitude'].values - minlatitude) / difflatitude)

dataset["Xcoord"] = Xcoord
dataset["Ycoord"] = Ycoord

dayofweek = dataset['day'].values % 7
dataset["dayOfWeek"] = dayofweek

del Xcoord, Ycoord, dayofweek
del col, difflatitude, difflongitude, i, minlatitude, minlongitude, row
del uniquelongitude, uniquelatitude

dataset.to_csv(r'dataset.csv', index  = False)




res = {col:dataset[col].value_counts() for col in dataset.columns}


#Test duplicate with different demand -------------------
df = dataset.drop(columns=['demand'])
duplicateRowsDF = df[df.duplicated()]
#-------------------------------------------------------




#Plot to image ----------------------------
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

#imgseries = np.dstack(imgseries)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(imgseries[:,:,41])




d1t0 = np.zeros(shape=(numlatitude,numlongitude))
day1 = dataset[(dataset["day"].values == 1)]
day1time0 = day1[(day1["normalizedTime"].values == 0)]
for i in range(len(day1time0)):
    X = day1time0["Xcoord"].values[i]
    Y = day1time0["Ycoord"].values[i]
    d1t0[int(Y)][int(X)] = day1time0["demand"].values[i] * 255
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('day1time0')
plt.imshow(d1t0)

import png
s = [[int(c) for c in row] for row in d1t0]
w = png.Writer(len(s[0]), len(s), greyscale=True)
f = open('d1t0.png', 'wb')
w.write(f, s)
f.close()
#---------------------------------------------   


# Plotting

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

day1 = dataset[(dataset["day"].values == 1.0)]
day1time1 = day1[(day1["normalizedTime"].values == 0)]
ax.scatter(day1time1["latitude"].values, day1time1["longitude"].values, day1time1["demand"].values, c='r', marker='o')
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Demand')
plt.show()


geo1 = dataset[(dataset["geohash6"]=="qp03wc")]
plt.scatter(geo1["normalizedDayTime"].values, geo1["demand"].values, color = 'red')

geo1 = dataset[(dataset["geohash6"]=="qp03wf")]
geo1day1 = geo1[(geo1["day"].values == 1.0)]
plt.scatter(geo1day1["normalizedTime"].values, geo1day1["demand"].values, color = 'red')

geo1day4 = geo1[(geo1["day"].values == 4.0)]
plt.scatter(geo1day4["normalizedTime"].values, geo1day4["demand"].values, color = 'blue')

geo1day8 = geo1[(geo1["day"].values == 8.0)]
plt.scatter(geo1day8["normalizedTime"].values, geo1day8["demand"].values, color = 'green')

geo1day11 = geo1[(geo1["day"].values == 11.0)]
plt.scatter(geo1day11["normalizedTime"].values, geo1day11["demand"].values, color = 'black')

geo1day15 = geo1[(geo1["day"].values == 15.0)]
plt.scatter(geo1day15["normalizedTime"].values, geo1day15["demand"].values, color = 'yellow')

geo1day18 = geo1[(geo1["day"].values == 18.0)]
plt.scatter(geo1day18["normalizedTime"].values, geo1day18["demand"].values, color = 'purple')

plt.title('qp03wc day 1-4')
plt.xlabel('Normalized Time')
plt.ylabel('Demand')
plt.show()


geo2 = dataset[(dataset["geohash6"]=="qp03pn")]
geo2day1 = geo2[(geo2["day"].values == 1.0)]
plt.scatter(geo1day1["normalizedTime"].values, geo1day1["demand"].values, color = 'red')

geo2day2 = geo2[(geo2["day"].values == 2.0)]
plt.scatter(geo2day2["normalizedTime"].values, geo2day2["demand"].values, color = 'blue')

geo2day3 = geo2[(geo2["day"].values == 3.0)]
plt.scatter(geo2day3["normalizedTime"].values, geo2day3["demand"].values, color = 'green')

geo2day4 = geo2[(geo2["day"].values == 4.0)]
plt.scatter(geo2day4["normalizedTime"].values, geo2day4["demand"].values, color = 'black')

plt.title('qp03pn day 1-4')
plt.xlabel('Normalized Time')
plt.ylabel('Demand')
plt.show()


# Plotting End -------------------------------------------------



dataset_train = dataset[~((dataset["day"].values == 61) & (dataset["normalizedTime"].values > 90))]
dataset_test = dataset[((dataset["day"].values == 61) & (dataset["normalizedTime"].values > 90))]

X_train = dataset_train[["latitude","longitude","day","normalizedTime"]].values
y_train = dataset_train["demand"].values

X_test = dataset_test[["latitude","longitude","day","normalizedTime"]].values
y_test = dataset_test["demand"].values









#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)




y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



import keras
from keras.models import Sequential
from keras.layers import Dense

reg = Sequential()

reg.add(Dense(units = 4, kernel_initializer='uniform', activation='relu', input_dim = 4))

reg.add(Dense(units = 4, kernel_initializer='uniform', activation='relu'))

reg.add(Dense(units = 1, kernel_initializer='uniform'))

reg.compile(optimizer="adam", loss="mse")


reg.fit(X_train, y_train, batch_size=100, epochs=100)
y_pred = reg.predict(X_test)



reg = LinearRegression()
reg = SVR(gamma='scale', C=1.0, epsilon=0.2)
reg = PolynomialFeatures(degree = 4)
reg = RandomForestRegressor(n_estimators = 250, random_state = 0, min_samples_split = 6)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)



r2 = r2_score(y_test, y_pred)  
r2








#test CNN LSTM

# Creating a data structure with 95 timesteps and 1 output
X_train = []
y_train = []
for i in range(96, imgseries.shape[2]):
    X_train.append(imgseries[:,:,i-96:i])
    y_train.append(imgseries[:,:,i])
X_train, y_train = np.array(X_train), np.array(y_train)

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import InputLayer


model = Sequential()

model.add(TimeDistributed(Conv2D(32,(3,3), activation = 'relu'),input_shape=(95,46,36,1))) 
print (model.output_shape)

model.add(TimeDistributed(MaxPooling2D(pool_size = (2, 2))))
print (model.output_shape)

model.add(TimeDistributed(Conv2D(32,(3,3), activation = 'relu'))) 
print (model.output_shape)

model.add(TimeDistributed(MaxPooling2D(pool_size = (2, 2))))
print (model.output_shape)

model.add(TimeDistributed(Flatten()))
print (model.output_shape)

model.add(LSTM(units = 50, return_sequences = True))
print (model.output_shape)

model.add(Dropout(0.2))
print (model.output_shape)

model.add(LSTM(units = 50, return_sequences = True))
print (model.output_shape)

model.add(Dropout(0.2))
print (model.output_shape)

model.add(LSTM(units = 50, return_sequences = True))
print (model.output_shape)

model.add(Dropout(0.2))
print (model.output_shape)

model.add(LSTM(units = 50))
print (model.output_shape)

model.add(Dropout(0.2))
print (model.output_shape)

model.add(Dense(units = 1656))
print (model.output_shape)

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mse')

# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)



#test Conv-LSTM

X_train = []
y_train = []
length = imgseries.shape[0]
#length = 4000
for i in range(48, length):
    X_train.append(imgseries[i-48:i,:,:,:])
    y_train.append(imgseries[i-47:i+1,:,:,:])
X_train = np.array(X_train)
y_train = np.array(y_train)




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

seq.fit(X_train, y_train, batch_size=4,
        epochs=50, validation_split=0.05)

seq.save('conv_lstm_time48_filter32_batch4.h5') 

# returns a compiled model
# identical to the previous one
seq = load_model('conv_lstm_time48_filter32_batch4.h5')


which = len(X_train)-1
X_test = X_train[len(X_train)-1][::, ::, ::, ::]

new_pos = seq.predict(X_test[np.newaxis, ::, ::, ::, ::])
new = new_pos[::, -1, ::, ::, ::]



col_names =  ['latitude', 'longitude', 'demand_prediction']
newDF = pd.DataFrame()
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

Xcoord = int((dataset['longitude'].values - minlongitude) / difflongitude)
Ycoord = int((dataset['latitude'].values - minlatitude) / difflatitude)

for i in range(new.shape[2]):
    latitude = minlatitude + difflatitude * i
    for j in range(new.shape[3]):
        longitude = minlongitude + difflongitude * j
        pred = {'latitude':latitude, 'longitude': longitude, 'demand_prediction': new[i][j]}
        
        

        
        
        
        
        
        