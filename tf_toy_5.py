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


def dataframe_optimizer(df):

    # Convert in np_array
    np_df = df.to_numpy()

    new_df_np = np.zeros((np_df.shape[0], 66))
    for i in range(0, np_df.shape[0]):

        sender = df['sender'][i]
        new_df_np[i][0] = sender
        sender_position_x = df['x_{}'.format(sender)][i]
        sender_position_y = df['y_{}'.format(sender)][i]
        new_df_np[i][1] = sender_position_x
        new_df_np[i][2] = sender_position_y

        start_idx_A = 1
        end_idx_A = 11
        start_idx_B = 11
        end_idx_B = 22
        if sender >= 11:
            start_idx_A = 11
            end_idx_A = 22
            start_idx_B = 1
            end_idx_B = 11
        # Add friedns position
        idx = 3
        # Store distance between each other gamer in the same time:
        dist = []
        for j in range(start_idx_A, end_idx_A):
            if i != sender:
                player_x = df['x_{}'.format(j)][i]
                new_df_np[i][idx] = player_x
                idx += 1
                player_y = df['y_{}'.format(j)][i]
                new_df_np[i][idx] = player_y
                idx += 1
                distance = np.sqrt((sender_position_x - player_x) ** 2 + (sender_position_y - player_y) ** 2)
                dist.append(distance)
        # Ad enemies position
        for j in range(start_idx_B, end_idx_B):
            player_x = df['x_{}'.format(j)][i]
            new_df_np[i][idx] = player_x
            idx += 1
            player_y = df['y_{}'.format(j)][i]
            new_df_np[i][idx] = player_y
            idx += 1
            distance = np.sqrt((sender_position_x - player_x) ** 2 + (sender_position_y - player_y) ** 2)
            dist.append(distance)
        # Aajout des distances par rapport aux amis et ennemis
        for j in range(0, len(dist)):
            new_df_np[i][idx] = dist[j]
            idx += 1

    for i in range(0, new_df_np.shape[1]):
        print(new_df_np[0][i])

    return new_df_np




if __name__ == '__main__':

    # Import dataset
    input_dataframe = pd.read_csv('input_training_set.csv', sep=',')
    output_dataframe = pd.read_csv('output_training_set.csv', sep=',')
    target = output_dataframe.to_numpy().reshape((-1, 1))
    # Optimize dataframe:
    inpt = dataframe_optimizer(input_dataframe)

    # Make a target array who said if pass was intercept (1) or not (0)
    target_intercept = np.zeros(target.shape)
    for i in range(0, target_intercept.shape[0]):
        #print('{} - {}'.format(input[i][0], target[i][0]))
        sender_team = 0
        if inpt[i][0] > 11:
            sender_team = 1
        receiver_team = 0
        if target[i][0] > 11:
            receiver_team = 1
        if sender_team != receiver_team:
            target_intercept[i][0] = 1


    # Create the model:
    model = tf.keras.models.Sequential()

    # Add layers
    model.add(tf.keras.layers.Dense(inpt.shape[1], activation='tanh'))
    model.add(tf.keras.layers.Dense(inpt.shape[1], activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))

    # COmpile the model:
    model.compile(
        loss='binary_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )

    # Fit
    history = model.fit(inpt, target_intercept, epochs=100, validation_split=0.2)



