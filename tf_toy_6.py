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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils






class network():

    def __init__(self):

        # Create the model:
        self.model = tf.keras.models.Sequential()

        # Add layers
        self.model.add(tf.keras.layers.Dense(60, activation='relu'))
        self.model.add(tf.keras.layers.Dense(60, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='tanh'))

        # Loss object: compute the error
        self.train_loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.test_loss_object = tf.keras.losses.CategoricalCrossentropy()
        # Optimizer: optimize parameters to decrease error
        self.optimizer = tf.keras.optimizers.Adam()
        # Accumulator: Metrics use to track the progress of the training loss during the training
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    @tf.function
    def train_step(self, x_train, y_train, x_test, y_test):
        """
        Training function
        """
        # Find gradient:
        with tf.GradientTape() as tape:     # To capture errors for the gradient modification
            # Make prediction
            train_predictions = self.model(x_train)
            test_predictions = self.model(x_test)
            # Get the error:
            train_loss = self.train_loss_object(y_train, train_predictions)
            test_loss = self.test_loss_object(y_test, test_predictions)

        # Compute the gradient who respect the loss
        gradients = tape.gradient(train_loss, self.model.trainable_variables)
        # Change weights of the model
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # Add error in accumulator
        self.train_loss(train_loss)
        self.test_loss(test_loss)

    def personal_loss(self, y, x):

        total = 0
        for i in range(0, x.shape[0]):

            predict_index = np.zeros(dtype=np.float32)
            best_predict = 0
            for j in range(0, len(x)):
                if x[i][j] >= best_predict:
                    best_predict = x[i][j]
                    predict_index = j
            if predict_index == y[i]:
                total += 1
        total /= x.shape[0]
        return total


    def train(self, x_train, y_train, x_test, y_test):

        for epoch in range(0, 10):
            for _ in range(0, 100):
                # Make a train step
                self.train_step(x_train, y_train, x_test, y_test)

            # Print the loss: return the mean of all error in the accumulator
            print('Test Loss: %s' % self.test_loss.result())
            print('Train Loss: %s' % self.train_loss.result())
            # Reset the accumulator
            self.train_loss.reset_states()
            self.test_loss.reset_states()


def distance(a, b):

    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def make_pair_of_players(X_, y_=None):
    n_ = X_.shape[0]
    pair_feature_col = ["sender", "x_sender", "y_sender", "player_j", "x_j", "y_j", "same_team"]
    X_pairs = pd.DataFrame(data=np.zeros((n_ * 22, len(pair_feature_col))), columns=pair_feature_col)
    y_pairs = pd.DataFrame(data=np.zeros((n_ * 22, 1)), columns=["pass"])

    # From pass to pair of players
    idx = 0
    for i in range(n_):
        sender = X_.iloc[i].sender
        players = np.arange(1, 23)
        # other_players = np.delete(players, sender-1)
        p_i_ = X_.iloc[i]
        for player_j in players:

            X_pairs.iloc[idx] = [sender, p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)],
                                 player_j, p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)],
                                 same_team_(sender, player_j)]

            if not y_ is None:
                y_pairs.iloc[idx]["pass"] = int(player_j == y_.iloc[i])
            idx += 1

    return X_pairs, y_pairs

def same_team_(sender,player_j):
    if sender <= 11:
        return int(player_j <= 11)
    else:
        return int(player_j > 11)


def first():


    # Import dataset
    input_dataframe = pd.read_csv('input_training_set.csv', sep=',')
    output_dataframe = pd.read_csv('output_training_set.csv', sep=',')

    inpt = input_dataframe.to_numpy()[:, :-1]
    target = output_dataframe.to_numpy().reshape((-1, 1))

    # Update player index
    for i in range(0, inpt.shape[0]):
        inpt[i][0] = inpt[i][0] - 1
        target[i][0] = target[i][0] - 1

    # Make a target array who said if pass was intercept (1) or not (0)
    target_intercept = np.zeros(target.shape)
    for i in range(0, target_intercept.shape[0]):
        sender_team = 0
        if inpt[i][0] >= 11:
            sender_team = 1
        receiver_team = 0
        if target[i][0] >= 11:
            receiver_team = 1
        if sender_team != receiver_team:
            target_intercept[i][0] = 1

    # Normalizer
    for i in range(2, inpt.shape[1]):
        if i % 2 == 0:
            for j in range(0, inpt.shape[0]):
                inpt[j][i] = inpt[j][i] / 5250
        else:
            for j in range(0, inpt.shape[0]):
                inpt[j][i] = inpt[j][i] / 3400
    for i in range(0, inpt.shape[0]):
        inpt[i][1] = inpt[i][1] / 3000000

    # One hot encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(target)

    new_target = np_utils.to_categorical(integer_encoded)

    for i in range(0, 20):
        print(new_target[i, :])


    # Create the model:
    mdl = network()

    # Split train and validation
    x_train, x_test, y_train, y_test = train_test_split(inpt, new_target, test_size=0.2)

    # Train the model
    mdl.train(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    print('test')
    # Import dataset
    input_dataframe = pd.read_csv('input_training_set.csv', sep=',')
    output_dataframe = pd.read_csv('output_training_set.csv', sep=',')

    x_pairs, y_pairs = make_pair_of_players(input_dataframe, output_dataframe)

    # Save to a csv file
    x_pairs.to_csv('save_x_pairs.csv', header=True, index=True, mode='w')
    y_pairs.to_csv('save_y_pairs.csv', header=True, index=True, mode='w')

    print(x_pairs)














