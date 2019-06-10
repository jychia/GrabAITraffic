import geohash
import numpy as np
import pandas as pd
import os

def preprocessing_rawdata(file_name):

    print("Start reading raw data")

    dataset = pd.read_csv(file_name, quoting = 3)

    print("Decoding geohash")

    longitude = []
    latitude = []

    for g in dataset["geohash6"].values:
        la, lo = geohash.decode(g)
        latitude.append(la)
        longitude.append(lo)
    
    dataset["latitude"] = latitude
    dataset["longitude"] = longitude
    
    print("Calculating normalized day and time")

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
    
    print("Calculating X and Y coordinates")
    
    uniquelatitude = dataset['latitude'].unique().tolist()
    uniquelatitude.sort()
    difflatitude = uniquelatitude[1] - uniquelatitude[0]
    minlatitude = uniquelatitude[0]
    
    uniquelongitude = dataset['longitude'].unique().tolist()
    uniquelongitude.sort()
    difflongitude = uniquelongitude[1] - uniquelongitude[0]
    minlongitude = uniquelongitude[0]
    
    Xcoord = (dataset['longitude'].values - minlongitude) / difflongitude
    Ycoord = (dataset['latitude'].values - minlatitude) / difflatitude
    
    dataset["Xcoord"] = Xcoord
    dataset["Ycoord"] = Ycoord
    
    edited_file_name = os.path.splitext(file_name)[0] + "_edited.csv"
    
    print("Finish preprocessing, saving data into ", edited_file_name)
    
    dataset.to_csv(edited_file_name, index  = False)
    
    return edited_file_name


def plot_data_to_img(dataset):
    
    img_series = []
    colNum = int(dataset["Ycoord"].values.max()+1)
    rowNum = int(dataset["Xcoord"].values.max()+1)
    
    print("Plotting data into images")
    
    for d in range(dataset["day"].values.min(), dataset["day"].values.max()+1):
        for t in range(int(dataset["normalizedTime"].values.min()), int(dataset["normalizedTime"].values.max()+1)):
            img = np.zeros(shape=(colNum,rowNum,1))
            day = dataset[(dataset["day"].values == d)]
            daytime = day[(day["normalizedTime"].values == t)]
        
            for i in range(len(daytime)):
                X = daytime["Xcoord"].values[i]
                Y = daytime["Ycoord"].values[i]
                img[int(Y)][int(X)][0] = daytime["demand"].values[i]
            img_series.append(img)
    img_series = np.array(img_series)
    
    return img_series
