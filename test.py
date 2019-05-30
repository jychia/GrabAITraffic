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
    b = dataset["day"][i] * 96 + a
    normalizedDayTime.append(b)

dataset["normalizedTime"] = normalizedTime
dataset["normalizedDayTime"] = normalizedDayTime

del normalizedTime, normalizedDayTime, time, a, b





dataset.to_csv(r'dataset.csv')




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

geo1 = dataset[(dataset["geohash6"]=="qp03wc")]
geo1day1 = geo1[(geo1["day"].values == 1.0)]
plt.scatter(geo1day1["normalizedTime"].values, geo1day1["demand"].values, color = 'red')

geo1day2 = geo1[(geo1["day"].values == 2.0)]
plt.scatter(geo1day2["normalizedTime"].values, geo1day2["demand"].values, color = 'blue')

geo1day3 = geo1[(geo1["day"].values == 3.0)]
plt.scatter(geo1day3["normalizedTime"].values, geo1day3["demand"].values, color = 'green')

geo1day4 = geo1[(geo1["day"].values == 4.0)]
plt.scatter(geo1day4["normalizedTime"].values, geo1day4["demand"].values, color = 'black')

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
sc2 = StandardScaler()
y_train = sc2.fit_transform(y_train)
y_test = sc2.transform(y_test)

reg = LinearRegression()
reg = SVR(gamma='scale', C=1.0, epsilon=0.2)
reg = PolynomialFeatures(degree = 4)
reg = RandomForestRegressor(n_estimators = 250, random_state = 0, min_samples_split = 6)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

r2 = r2_score(y_test, y_pred)  
r2

