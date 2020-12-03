# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager

from pandas import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

MAX_X_ABS_VAL = 5250
MAX_Y_ABS_VAL = 3400





if __name__ == '__main__':

    # Import dataset
    dataframe = pd.read_csv('input_training_set.csv', sep=',')
    data_np = dataframe.to_numpy(dtype=np.float64)
    print("---test---")
    print(data_np.shape)

    output_dataframe = pd.read_csv('output_training_set.csv', sep=',')
    output_np = output_dataframe.to_numpy()
    
    print(data_np[0])
    rest = np.delete(data_np, [0, 1], 1)
    print(rest[0])
    sender_ts = data_np[:, :2]
    scaler = StandardScaler()
    data_np = scaler.fit_transform(data_np)
    # data_np = np.hstack((sender_ts, rest))
    print(data_np[0])

    # Add output column
    dataframe.insert(data_np.shape[1], 'output', output_np)

    # Create the model:
    model = tf.keras.models.Sequential()

    # Add layers
    model.add(tf.keras.layers.Dense(47, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(22, activation='softmax'))


    # COmpile the model:
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(output_np)
    y_data=np_utils.to_categorical(integer_encoded)
    x = data_np[:, :46]
    print("---test---")
    print(y_data.shape)
    print(x.shape)

    # Fit
    history = model.fit(x, y_data, epochs=100, validation_split=0.2)

    pred = model.predict(data_np[0, :46].reshape((1, 46)))
    score, acc = model.evaluate(x[5000:], y_data[5000:], verbose=0)

    print('Test score:', score)
    print('Test acc:', acc)

    model.summary()







