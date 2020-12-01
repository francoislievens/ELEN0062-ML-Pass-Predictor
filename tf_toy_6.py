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


METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]



class network():

    def __init__(self):

        # Create the model:
        self.model = tf.keras.models.Sequential()

        # Add layers
        self.model.add(tf.keras.layers.Dense(60, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(60, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(60, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(60, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Loss object: compute the error
        self.train_loss_object = tf.keras.losses.BinaryCrossentropy()
        self.test_loss_object = tf.keras.losses.BinaryCrossentropy()
        # Optimizer: optimize parameters to decrease error
        self.optimizer = tf.keras.optimizers.Adam()
        # Accumulator: Metrics use to track the progress of the training loss during the training
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    @tf.function
    def train_step(self, x_train, y_train, x_test, y_test, s_weights_train, s_weights_test):
        """
        Training function
        """
        # Find gradient:
        with tf.GradientTape() as tape:     # To capture errors for the gradient modification
            # Make prediction
            train_predictions = self.model(x_train)
            # Get the error:
            train_loss = self.train_loss_object(y_train, train_predictions, sample_weight=s_weights_train)

        test_predictions = self.model(x_test)
        test_loss = self.test_loss_object(y_test, test_predictions, sample_weight=s_weights_test)
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


    def train(self, x_train, y_train, x_test, y_test, s_weights_train, s_weights_test):

        for epoch in range(0, 100):
            for _ in range(0, 100):
                # Make a train step
                self.train_step(x_train, y_train, x_test, y_test, s_weights_train, s_weights_test)

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
    pair_feature_col = ["sender", "x_sender", "y_sender", "player_j", "x_j", "y_j", "same_team",
                        "same_team_moy_dist", "adv_team_moy_dist", "closer_same_team_dist",
                        "closer_adv_dist", "game_zone"]
    X_pairs = pd.DataFrame(data=np.zeros((n_ * 22, len(pair_feature_col))), columns=pair_feature_col)
    y_pairs = pd.DataFrame(data=np.zeros((n_ * 22, 1)), columns=["pass"])

    # From pass to pair of players
    idx = 0
    for i in range(n_):
        sender = X_.iloc[i].sender
        players = np.arange(1, 23)
        # other_players = np.delete(players, sender-1)
        p_i_ = X_.iloc[i]
        # Mean dist from same team
        s_t_dist = 0
        # Mean dist adv
        adv_t_dist = 0
        idx_start = idx
        # Closer same team dist
        closer_st = 5000
        # Closer adv dist
        closer_adv = 5000
        # Game zone
        game_zone = 0
        if X_.iloc[i]["x_{:0.0f}".format(sender)] < 0:
            game_zone = 1
        for player_j in players:

            X_pairs.iloc[idx] = [sender, p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)],
                                 player_j, p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)],
                                 same_team_(sender, player_j), 0, 0, 0, 0, 0]
            if same_team_(sender, player_j) == 1:
                s_t_dist += distance((p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)]), (p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)]))
                if s_t_dist <= closer_st:
                    closer_st = s_t_dist
            if same_team_(sender, player_j) == 0:
                adv_t_dist += distance((p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)]), (p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)]))
                if adv_t_dist <= closer_adv:
                    closer_adv = adv_t_dist
            if not y_ is None:
                y_pairs.iloc[idx]["pass"] = int(player_j == y_.iloc[i])
            idx += 1
        idx_end = idx
        s_t_dist /= 10
        adv_t_dist /= 11
        for j in range(idx_start, idx_end):
            X_pairs.iloc[j]['same_team_moy_dist'] = s_t_dist
            X_pairs.iloc[j]['adv_team_moy_dist'] = adv_t_dist
            X_pairs.iloc[j]['closer_same_team_dist'] = closer_st
            X_pairs.iloc[j]['closer_adv_dist'] = closer_adv
            X_pairs.iloc[j]['game_zone'] = game_zone


    return X_pairs, y_pairs

def same_team_(sender,player_j):
    if sender <= 11:
        return int(player_j <= 11)
    else:
        return int(player_j > 11)

def import_and_save_dataset():

    # Import dataset
    input_dataframe = pd.read_csv('input_training_set.csv', sep=',')
    output_dataframe = pd.read_csv('output_training_set.csv', sep=',')

    x_pairs, y_pairs = make_pair_of_players(input_dataframe, output_dataframe)

    # Save to a csv file
    x_pairs.to_csv('save_x_pairs.csv', header=True, index=True, mode='w')
    y_pairs.to_csv('save_y_pairs.csv', header=True, index=True, mode='w')


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

    #import_and_save_dataset()

    # Read dataset:
    x = pd.read_csv('save_x_pairs.csv', sep=',', index_col=0)
    y = pd.read_csv('save_y_pairs.csv', sep=',', index_col=0)

    # Got numpy versions
    x_np = x.to_numpy()
    y_np = y.to_numpy()

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x_np, y_np, test_size=0.2)

    # Create the model
    model = network()

    # One hot encoding
    y_train_np = np.copy(y_train)
    x_train_np = np.copy(x_train)
    y_test_np = np.copy(y_test)
    x_test_np = np.copy(x_test)
    one_hot_target_train = np.zeros((x_train_np.shape[0], 2))
    one_hot_target_test = np.zeros((x_test_np.shape[0], 2))
    for i in range(0, x_train_np.shape[0]):
        if y_train_np[i] == 1:
            one_hot_target_train[i][1] = 1
        else:
            one_hot_target_train[i][0] = 1
    for i in range(0, x_test_np.shape[0]):
        if y_test_np[i] == 1:
            one_hot_target_test[i][1] = 1
        else:
            one_hot_target_test[i][0] = 1

    # Sample weights
    sample_weights_train = np.ones(x_train.shape[0])
    sample_weights_test = np.ones(x_test.shape[0])
    for i in range(0, x_train.shape[0]):
        if y_train[i] == 1:
            sample_weights_train[i] *= 22
    for i in range(0, x_test.shape[0]):
        if y_test[i] == 1:
            sample_weights_test[i] *= 22


    # Train the model
    model.train(x_train, y_train, x_test, y_test, sample_weights_train, sample_weights_test)

    pred = model.model.predict(x_test)

    for i in range(0, 10):
        print(pred[i])

    accu = 0

    one_counter = 0
    true_one_counter = 0
    for i in range(0, pred.shape[0]):
        if y_test[i] == 1: one_counter += 1
        if pred[i] >= 0.5 and y_test[i] == 1:
            accu += 1
            true_one_counter += 1
        if pred[i] < 0.5 and y_test[i] == 0:
            accu += 1




    accu /= pred.shape[0]

    print('Test accuracy: {}'.format(accu))
    print('One counter = {} - True one counter = {}'.format(one_counter, true_one_counter))

















