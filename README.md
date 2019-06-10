# Grab AI for S.E.A.: Traffic Management Challenge
by Jia Yuan Chia

This repository is a submission for Grab AI for SEA challenge: [Traffic Management Challenge](https://www.aiforsea.com/traffic-management).

The competition requires participants to build a model trained on a historical demand dataset, that can forecast demand on a Hold-out test dataset. 
The model should be able to accurately forecast ahead by T+1 to T+5 time intervals (where each interval is 15-min) given all data up to time T.

## Model Concept and Explanation
In the provided training data on booking demand, the raw data are being preprocessed into the following columns:

| Columns        | Description           |
| -------------- | --------------------- |
| geohash6       | Original provided dataset to represent location |
| day            | Original provided dataset to represent which consecutive days in dataset      |
| timestamp      | Original provided dataset to represent time of day      |
| demand         | Original provided dataset to represent booking demand of that location in that particular timestep and day |
| latitude       | Latitude value decoded from geohash      |
| longitude      | Longitude value decoded from geohash      |
| normalizedTime | Normalized timesteps in the day, where each value represents 15-min slice. (0 = 00:00, 1 = 00:15, 2 = 00:30, etc...) |
| normalizedDayTime | Normalized overall timesteps over 61 days, where each value represents 15-min slice. (0 = day 1 00:00, 1 = day 1 00:15, 96 = day 2 00:00, etc...) |
| Xcoord | Normalized X coordinates = ( longitudes - min value in all longitudes ) / each longitude step |
| Ycoord | Normalized Y coordinates = ( latitudes - min value in all latitudes ) / each latitude step |

The geohashes are able to be decoded into a total of 46 unique latitudes and 36 unique longitudes, where they can then be plotted into a 46 x 36 grid of traffic demand data.
To process the provided data, they were first splitted by their corresponding day and timesteps. 
Then, the traffic demands for each geohash on that specific day and timesteps are used to be plotted into an image of 46 x 36.

Sample visualization on booking demand plots:

![alt text](Media/day_1_12am.PNG)
![alt text](Media/day_5_3pm.PNG)

After plotting the data into the booking demand plots, there will be a total of 96 plots for each day (96 distinct timesteps), 
and a total of 5856 plots over 61 days.

These plots are then being fed into a 4 layered ConvLSTM2D neural network with 32 neurons each, where it is used to train the model for predictions of next plot.

To predict the demands for T+1 up to T+5, the model will first be used to predict a new booking demand plot for T+1, where it's input data consists of 96 previous demand plots (T-95 to T, equivalent to 1 day's data).
With that data, the model then predicts the T+1 demand plot.
From there, the T+1 plot is added into the input data and the T-95 plot is removed from the input data.
The input data is then used to predict T+2 booking demand plot.
The process is then repeated until T+5 is predicted.

## Folder Structure

.
├── Datasets
│   ├── data.sh
│   ├── dl_large_files.sh
│   ├── fastai.sh
│   ├── img.sh
│   ├── lgb.sh
│   └── setup_linux.sh
├── Media                   # Images used in README.md
│   ├── day_1_12am.PNG
│   ├── day_5_3pm.PNG
│   └── README_logo.PNG
├── Models                  # Saved models used for testing, 
│   ├── common
│   │   ├── __init__.py
│   │   ├── split_data.py
│   │   └── topic.py
│   ├── heuristic
│   │   ├── fashion_library.json
│   │   ├── __init__.py
│   │   └── mobile
│   │       ├── enricher.py
│   │       ├── extractor.py
│   │       └── __init__.py
│   ├── image
│   │   ├── fastai
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   └── ml_model.py
│   │   ├── __init__.py
│   │   └── pytorch
│   │       ├── dataset.py
│   │       ├── __init__.py
│   │       ├── test.py
│   │       ├── train.py
│   │       └── validation.py
└── README.md

