import geohash
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.metrics import r2_score
import preprocessing
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model



file_name = 'training.csv'

edited_file_name = preprocessing.preprocessing_rawdata(file_name)

dataset = pd.read_csv(edited_file_name, quoting = 3)

testing_img = preprocessing.plot_data_to_img(dataset)

model_name = 'final_model.h5'
seq = load_model(model_name)

y_pred = []
timestep = 96
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


final_pred = pd.DataFrame(columns = ['latitude', 'longitude', 'geohash6', 'prediction', 'demand', 'TPlus'])
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
                                 'demand':[y_test[k][i][j]], \
                                 'TPlus':[k+1]})
            final_pred = final_pred.append(pred,ignore_index=True)
        
del uniquelatitude, difflatitude, minlatitude
del uniquelongitude, difflongitude, minlongitude
del i, j, k, pred


print("Finish post-processing prediction data, start evaluating")

r2 = r2_score(final_pred['demand'], final_pred['prediction'])  
print("R2 score for total = ", r2)
        
r2 = r2_score(y_test[0].flatten(), y_pred[0].flatten())  
print("R2 score for T+1 = ", r2)

r2 = r2_score(y_test[1].flatten(), y_pred[1].flatten())  
print("R2 score for T+2 = ", r2)

r2 = r2_score(y_test[2].flatten(), y_pred[2].flatten())  
print("R2 score for T+3 = ", r2)

r2 = r2_score(y_test[3].flatten(), y_pred[3].flatten())  
print("R2 score for T+4 = ", r2)

r2 = r2_score(y_test[4].flatten(), y_pred[4].flatten())  
print("R2 score for T+5 = ", r2)


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
