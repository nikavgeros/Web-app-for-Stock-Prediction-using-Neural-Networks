import os
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader.data import DataReader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU, SimpleRNN
from keras.layers import Dropout
from tensorflow.keras.optimizers import SGD

# Define Stock list
stocks = ['AAPL', 'AMZN', 'GE', 'GOOGL', 'IBM', 'MSFT', 'TSLA']
stock = "None"

# Define Paths
scaler_path = 'Scalers\\'
model_path = 'Models\\'
image_path = 'Images\\'

# Define parameters
timesteps = 60
start_date = '2016-01-01'
end_date = '2021-12-31'
split_date = '2021-10-31'


# Create paths
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, "One layer"))
        os.makedirs(os.path.join(path, "Two layers"))
        os.makedirs(os.path.join(path, "Three layers"))
        print("DIRECTORY CREATED: {}".format(os.path.abspath(path)))


# Load stock data
def load_data(stock):
    return DataReader(stock, 'yahoo', start_date, end_date)[["Close"]]


# Split data
def splitting_data(data, split_date):
    train = data[:split_date].values
    test = data[split_date:].values
    return train, test


# Scale data
def scale_data(data, file, save=True):
    sc = MinMaxScaler(feature_range = (0, 1))
    train_scaled = sc.fit_transform(data)
    if save:
        pickle.dump(sc, open(file, 'wb'))
    return train_scaled, sc


# Train timeseries to supervised
def train_to_supervised(data, start, end):
    X_train = []
    y_train = []
    for i in range(start, end):
        X_train.append(data[i-start:i, 0])
        y_train.append(data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train


# Test timeseries to supervised
def test_to_supervised(data, start, end, scaler):
    data = data.reshape(-1,1)
    data = scaler.transform(data)
    X_test = []
    for i in range(start, end):
        X_test.append(data[i-start:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test


# Build NN with one layer
def build_one_layer_nn(X, y, model, first_layer_units, file="One layer NN.h5", dropout=0.2, epochs=50, batch_size=32, optimizer='adam', loss='mean_squared_error', verbose=2, save=True):
    nn = Sequential()
    nn.add(model(units=first_layer_units, input_shape=(X.shape[1],1)))
    nn.add(Dropout(dropout))
    nn.add(Dense(units=1))
    nn.compile(optimizer=optimizer, loss=loss)
    nn.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    if save:
        nn.save(file)
    return nn

# Build NN with two layers
def build_two_layers_nn(X, y, model, first_layer_units, second_layer_units, file="Two layers NN.h5", dropout=0.2, epochs=50, batch_size=32, optimizer='adam', loss='mean_squared_error', verbose=2, save=True):
    nn = Sequential()
    nn.add(model(units=first_layer_units, return_sequences = True, input_shape=(X.shape[1],1)))
    nn.add(Dropout(dropout))
    nn.add(SimpleRNN(units=second_layer_units))
    nn.add(Dropout(dropout))
    nn.add(Dense(units=1))
    nn.compile(optimizer=optimizer, loss=loss)
    nn.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    if save:
        nn.save(file)
    return nn


# Build NN with three layers
def build_three_layers_nn(X, y, model, first_layer_units, second_layer_units, third_layer_units, file="Three layers NN.h5", dropout=0.2, epochs=50, batch_size=32, optimizer='adam', loss='mean_squared_error', verbose=2, save=True):
    nn = Sequential()
    nn.add(model(units=first_layer_units, return_sequences = True, input_shape=(X.shape[1],1)))
    nn.add(Dropout(dropout))
    nn.add(SimpleRNN(units=second_layer_units, return_sequences = True,))
    nn.add(Dropout(dropout))
    nn.add(SimpleRNN(units=third_layer_units))
    nn.add(Dropout(dropout))
    nn.add(Dense(units=1))
    nn.compile(optimizer=optimizer, loss=loss)
    nn.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    if save:
        nn.save(file)
    return nn


# Predictions
def make_predictions(data, model, scaler="None"):
    predictions = model.predict(data)
    if scaler == "None":
        return predictions
    else:
        return scaler.inverse_transform(predictions)


# Root Mean Square Error
def rmse_calculate(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))


# Visualization
def visualization(actual, predicted, title, file, save=True):
    plt.style.use('fivethirtyeight')
    plt.figure(figsize = (14,4))
    plt.plot(actual, color = 'red', label = f'Actual Closing Price')
    plt.plot(predicted, color = 'blue', label = f'Predicted Closing Price')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(f'Closing Price $')
    plt.legend()
    if save:
        plt.savefig(file)
