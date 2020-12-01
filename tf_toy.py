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

MAX_X_ABS_VAL = 5250
MAX_Y_ABS_VAL = 3400





if __name__ == '__main__':

    # Import dataset
    dataframe = pd.read_csv('input_training_set.csv', sep=',')
    data_np = dataframe.to_numpy(dtype=np.float64)

    output_dataframe = pd.read_csv('output_training_set.csv', sep=',')
    output_np = output_dataframe.to_numpy()
    for i in range(0, len(output_np)):
        output_np[i] -= 1
    print(output_np)

    # Add output column
    dataframe.insert(data_np.shape[1], 'output', output_np)

    # Create the model:
    model = tf.keras.models.Sequential()

    # Add layers
    model.add(tf.keras.layers.Dense(47, activation='relu'))
    model.add(tf.keras.layers.Dense(47, activation='relu'))
    model.add(tf.keras.layers.Dense(22, activation='softmax'))

    output = model.predict(data_np[0, :46].reshape((1, 46)))
    print(output)
    print(output.shape)

    # model.summary()

    # COmpile the model:
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )

    # Fit
    history = model.fit(data_np[:, :46].reshape((-1, 46)), output_np.reshape((-1, 1)), epochs=10)






