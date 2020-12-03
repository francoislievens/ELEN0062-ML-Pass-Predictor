# -*- coding: utf-8 -*-

import os
import time
import datetime
from contextlib import contextmanager

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

MAX_X_ABS_VAL = 5250
MAX_Y_ABS_VAL = 3400





if __name__ == '__main__':

    # Import dataset
    dataframe = pd.read_csv('input_training_set.csv', sep=',')
    data_np = dataframe.to_numpy(dtype=np.float64)
    for i in range(0, data_np.shape[0]):
        data_np[i][0] = data_np[i][0] - 1

    output_dataframe = pd.read_csv('output_training_set.csv', sep=',')
    data_output_np = output_dataframe.to_numpy()
    for i in range(0, len(data_output_np)):
        data_output_np[i] -= 1
    print(data_np)
    input = np.copy(data_np[:, :46])
    input.reshape(-1, 46)
    input.astype(float)
    target = np.copy(data_output_np)
    target.reshape(-1, 1)

    # Use standard scaller Normalizer
    scaler = StandardScaler()
    input = scaler.fit_transform(input)

    # Create the model:
    model = tf.keras.models.Sequential()

    # Add layers
    #model.add(tf.keras.layers.Dense(47, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(22, activation='softmax'))

    # COmpile the model:
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )

    # Fit
    history = model.fit(input, target, epochs=50)








