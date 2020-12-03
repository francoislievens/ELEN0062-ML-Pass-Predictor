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
    input_dataframe = pd.read_csv('input_training_set.csv', sep=',')
    inpt = input_dataframe.to_numpy(dtype=np.float64)[:, :46]
    # Adapt player index
    for i in range(0, inpt.shape[0]):
        inpt[i][0] = inpt[i][0] - 1

    # Add distance between players
    dist = np.zeros((inpt.shape[0], 22))
    for i in range(0, inpt.shape[0]):
        for j in range(0, 11):
            delta_x = (inpt[i][ int(inpt[i][0])*2 +2] - inpt[i][j*2 + 2]) ** 2
            delta_y = (inpt[i][ int(inpt[i][0])*2 +3] - inpt[i][j*2 + 3]) ** 2
            dist[i][j] = np.sqrt(delta_y + delta_x)

    # Add new columns to dataframe:
    for i in range(0, dist.shape[1]):
        input_dataframe.insert(inpt.shape[1] + i, 'sender_dist_{}'.format(i+1), dist[:, i])

    inpt = input_dataframe.to_numpy()[:, :-2]


    # Add distance from the center of the game = savoir si le joueur est en position attaque ou défense
    dist = np.zeros((inpt.shape[0], 1))
    for i in range(0, inpt.shape[0]):
        dist_from_center = inpt[i][int(inpt[i][0])]
        if inpt[i][0] >= 11:
            dist_from_center *= -1
    # Add the new column:
    input_dataframe.insert(inpt.shape[1], 'dist_from_center', dist[:, 0])
    inpt = input_dataframe.to_numpy()[:, :-1]


    output_dataframe = pd.read_csv('output_training_set.csv', sep=',')
    target = output_dataframe.to_numpy().reshape((-1, 1))
    for i in range(0, len(target)):
        target[i] -= 1

    # Make a target array who said if pass was intercept (1) or not (0)

    target_intercept = np.zeros(target.shape)
    for i in range(0, target_intercept.shape[0]):
        #print('{} - {}'.format(input[i][0], target[i][0]))
        sender_team = 0
        if inpt[i][0] >= 11:
            sender_team = 1
        receiver_team = 0
        if target[i][0] >= 11:
            receiver_team = 1
        if sender_team != receiver_team:
            target_intercept[i][0] = 1

    # Create the model:
    model = tf.keras.models.Sequential()

    # Add layers
    model.add(tf.keras.layers.Dense(inpt.shape[1], activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))

    # COmpile the model:
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )

    # Fit
    history = model.fit(inpt, target_intercept, epochs=10)








