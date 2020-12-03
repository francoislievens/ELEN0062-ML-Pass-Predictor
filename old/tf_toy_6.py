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
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as pw



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


MAX_X_ABS_VAL = 5250
MAX_Y_ABS_VAL = 3400

class network():

    def __init__(self):

        # Create the model:
        self.model = tf.keras.models.Sequential()

        # Add layers
        self.model.add(tf.keras.layers.Dense(7, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(7, activation='relu'))
        self.model.add(tf.keras.layers.Dense(4, activation='tanh'))
        #self.model.add(tf.keras.layers.Dropout(0.3))
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
    def train_step(self, x_train, y_train, x_test, y_test):
        """
        Training function
        """
        # Find gradient:
        with tf.GradientTape() as tape:     # To capture errors for the gradient modification
            # Make prediction
            train_predictions = self.model(x_train)
            # Get the error:
            train_loss = self.train_loss_object(y_train, train_predictions)

        test_predictions = self.model(x_test)
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

<<<<<<< HEAD:tf_toy_6.py
        for epoch in range(0, 50):
            for _ in range(0, 100):
=======
        for epoch in range(0, 10):
            for _ in range(0, 30):
>>>>>>> 11b1b8b94bdfab68b22e244aa3399d6dcd6b150d:old/tf_toy_6.py
                # Make a train step
                self.train_step(x_train, y_train, x_test, y_test)

            print('Epoch: {}'.format(epoch))
            # Print the loss: return the mean of all error in the accumulator
            print('Test Loss: %s' % self.test_loss.result())
            print('Train Loss: %s' % self.train_loss.result())
            # Reset the accumulator
            self.train_loss.reset_states()
            self.test_loss.reset_states()


def distance(a, b):

    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def is_pass_forward(sender, receiver, columns):
    sender_x = columns["x_{:0.0f}".format(sender)]
    receiver_x = columns["x_{:0.0f}".format(receiver)]
    left_most = MAX_X_ABS_VAL
    left_most_player = 0
    for player in range(1, 23):
        player_x = columns["x_{:0.0f}".format(player)]
        if left_most > player_x:
            left_most = player_x
            left_most_player = player
    same_team = 0
    if left_most_player < 12:
        if sender < 12:
            same_team = 1
    else:
        if sender >= 12:
            same_team = 1
    if same_team == 1:
        return np.abs(sender_x - receiver_x)/MAX_X_ABS_VAL if (receiver_x - left_most) > (sender_x - left_most) else -np.abs(sender_x - receiver_x)/MAX_X_ABS_VAL
    else:
        return np.abs(sender_x - receiver_x)/MAX_X_ABS_VAL if (receiver_x - left_most) < (sender_x - left_most) else -np.abs(sender_x - receiver_x)/MAX_X_ABS_VAL

def get_diagonale():
    return np.sqrt(MAX_X_ABS_VAL**2 + MAX_Y_ABS_VAL**2)

def compute_distance_(X_):
    d = np.zeros((X_.shape[0],))
    d = np.sqrt((X_["x_sender"]-X_["x_j"])**2 + (X_["y_sender"]-X_["y_j"])**2)
    return d


def min_dist_teammates(sender, receiver, columns):
    if sender == receiver:
        return 0
    sender_x = columns["x_{:0.0f}".format(sender)]
    sender_y = columns["y_{:0.0f}".format(sender)]
    receiver_x = columns["x_{:0.0f}".format(receiver)]
    receiver_y = columns["y_{:0.0f}".format(receiver)]
    dist_sender_receiver = np.sqrt((sender_x - receiver_x)**2 + (sender_y - receiver_y)**2)
    dist = get_diagonale()
    for player in range(1,23):
        if player == sender or player == receiver or same_team_(player, sender) != 1:
            continue
        player_x = columns["x_{:0.0f}".format(player)]
        player_y = columns["y_{:0.0f}".format(player)]
        dist_player_sender = np.sqrt((sender_x - player_x)**2 + (sender_y - player_y)**2)
        dist_player_receiver = np.sqrt((player_x - receiver_x)**2 + (player_y - receiver_y)**2)
        dist = min(dist, dist_player_sender)
    return dist



def avg_dist_teammates(sender, receiver, columns):
    if sender == receiver:
        return 0
    sender_x = columns["x_{:0.0f}".format(sender)]
    sender_y = columns["y_{:0.0f}".format(sender)]
    receiver_x = columns["x_{:0.0f}".format(receiver)]
    receiver_y = columns["y_{:0.0f}".format(receiver)]
    dist_sender_receiver = np.sqrt((sender_x - receiver_x)**2 + (sender_y - receiver_y)**2)
    dist = 0
    for player in range(1,23):
        if player == sender or player == receiver or same_team_(player, sender) != 1:
            continue
        player_x = columns["x_{:0.0f}".format(player)]
        player_y = columns["y_{:0.0f}".format(player)]
        dist_player_sender = np.sqrt((sender_x - player_x)**2 + (sender_y - player_y)**2)
        dist_player_receiver = np.sqrt((player_x - receiver_x)**2 + (player_y - receiver_y)**2)
        dist += dist_player_sender
    return dist

def avg_dist_opp(sender, receiver, columns):
    if sender == receiver:
        return 0
    sender_x = columns["x_{:0.0f}".format(sender)]
    sender_y = columns["y_{:0.0f}".format(sender)]
    receiver_x = columns["x_{:0.0f}".format(receiver)]
    receiver_y = columns["y_{:0.0f}".format(receiver)]
    dist_sender_receiver = np.sqrt((sender_x - receiver_x)**2 + (sender_y - receiver_y)**2)
    dist = 0
    for player in range(1,23):
        if player == sender or player == receiver or same_team_(player, sender) == 1:
            continue
        player_x = columns["x_{:0.0f}".format(player)]
        player_y = columns["y_{:0.0f}".format(player)]
        dist_player_sender = np.sqrt((sender_x - player_x)**2 + (sender_y - player_y)**2)
        dist_player_receiver = np.sqrt((player_x - receiver_x)**2 + (player_y - receiver_y)**2)
        dist += dist_player_receiver
    return dist/11


def min_dist_opp(sender, receiver, columns):
    if sender == receiver:
        return 0
    sender_x = columns["x_{:0.0f}".format(sender)]
    sender_y = columns["y_{:0.0f}".format(sender)]
    receiver_x = columns["x_{:0.0f}".format(receiver)]
    receiver_y = columns["y_{:0.0f}".format(receiver)]
    dist_sender_receiver = np.sqrt((sender_x - receiver_x)**2 + (sender_y - receiver_y)**2)
    dist = get_diagonale()
    for player in range(1,23):
        if player == sender or player == receiver or same_team_(player, sender) == 1:
            continue
        player_x = columns["x_{:0.0f}".format(player)]
        player_y = columns["y_{:0.0f}".format(player)]
        dist_player_sender = np.sqrt((sender_x - player_x)**2 + (sender_y - player_y)**2)
        dist_player_receiver = np.sqrt((player_x - receiver_x)**2 + (player_y - receiver_y)**2)
        dist = min(dist, dist_player_receiver)
    return dist


def min_dist_opp_sender(sender, receiver, columns):
    if sender == receiver:
        return 0
    sender_x = columns["x_{:0.0f}".format(sender)]
    sender_y = columns["y_{:0.0f}".format(sender)]
    receiver_x = columns["x_{:0.0f}".format(receiver)]
    receiver_y = columns["y_{:0.0f}".format(receiver)]
    dist_sender_receiver = np.sqrt((sender_x - receiver_x)**2 + (sender_y - receiver_y)**2)
    dist = get_diagonale()
    for player in range(1,23):
        if player == sender or player == receiver or same_team_(player, sender) == 1:
            continue
        player_x = columns["x_{:0.0f}".format(player)]
        player_y = columns["y_{:0.0f}".format(player)]
        dist_player_sender = np.sqrt((sender_x - player_x)**2 + (sender_y - player_y)**2)
        dist_player_receiver = np.sqrt((player_x - receiver_x)**2 + (player_y - receiver_y)**2)
        dist = min(dist, dist_player_sender)
    return dist

def max_cosine_similarity(sender, receiver, columns):
    if sender == receiver:
        return 0
    sender_x = columns["x_{:0.0f}".format(sender)]
    sender_y = columns["y_{:0.0f}".format(sender)]
    receiver_x = columns["x_{:0.0f}".format(receiver)]
    receiver_y = columns["y_{:0.0f}".format(receiver)]
    receiver_vector = [sender_x - receiver_x, sender_y - receiver_y]
    cos = -1
    for player in range(1,23):
        if player == sender or player == receiver or same_team_(player, sender) == 1:
            continue
        player_x = columns["x_{:0.0f}".format(player)]
        player_y = columns["y_{:0.0f}".format(player)]
        opponent_vector = [sender_x - player_x, sender_y - player_y]
        cos = max(cos, pw.cosine_similarity(receiver_vector, oppenent_vector))
    return cos



def make_pair_of_players(X_, y_=None):
    n_ = X_.shape[0]
    pair_feature_col = ["sender", "x_sender", "y_sender", "player_j", "x_j", "y_j", "same_team", "max_cs",
                        "is_pass_forward", "distance", "dist_opp_min", "dist_opp_min_sender", "dist_opp_avg", "dist_tm_min", "dist_tm_avg"]
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
            dist_opp_min = min_dist_opp(sender, player_j, p_i_)
            dist_opp_min_sender = min_dist_opp_sender(sender, player_j, p_i_)
            dist_opp_avg = avg_dist_opp(sender, player_j, p_i_)
            dist_tm_avg = avg_dist_teammates(sender, player_j, p_i_)
            dist_tm_min = min_dist_teammates(sender, player_j, p_i_)
            max_cs = max_cosine_similarity(sender, player_j, p_i_)
            X_pairs.iloc[idx] = [sender/22, p_i_["x_{:0.0f}".format(sender)]/MAX_X_ABS_VAL, p_i_["y_{:0.0f}".format(sender)]/MAX_Y_ABS_VAL,
                                 player_j/22, p_i_["x_{:0.0f}".format(player_j)]/MAX_X_ABS_VAL, p_i_["y_{:0.0f}".format(player_j)]/MAX_Y_ABS_VAL,
                                 same_team_(sender, player_j), max_cs, is_pass_forward(sender, player_j, p_i_), 0, dist_opp_min, dist_opp_min_sender, dist_opp_avg, dist_tm_min, dist_tm_avg]
            """
            if same_team_(sender, player_j) == 1:
                s_t_dist += distance((p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)]), (p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)]))
                if s_t_dist <= closer_st:
                    closer_st = s_t_dist
            if same_team_(sender, player_j) == 0:
                adv_t_dist += distance((p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)]), (p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)]))
                if adv_t_dist <= closer_adv:
                    closer_adv = adv_t_dist
            """
            if not y_ is None:
                y_pairs.iloc[idx]["pass"] = int(player_j == y_.iloc[i])
            
            idx += 1
        # idx_end = idx
        # s_t_dist /= 10
        # adv_t_dist /= 11
        # for j in range(idx_start, idx_end):
            # X_pairs.iloc[j]['same_team_moy_dist'] = s_t_dist
            # X_pairs.iloc[j]['adv_team_moy_dist'] = adv_t_dist
            # X_pairs.iloc[j]['closer_same_team_dist'] = closer_st
            # X_pairs.iloc[j]['closer_adv_dist'] = closer_adv
            # X_pairs.iloc[j]['game_zone'] = game_zone
    distance = compute_distance_(X_pairs)
    max_dist = np.max(distance)
    X_pairs["distance"] = distance / max_dist
    dist_opp_min = X_pairs["dist_opp_min"]
    max_dist_opp_min = np.max(X_pairs["dist_opp_min"])
    X_pairs["dist_opp_min"] = dist_opp_min / max_dist_opp_min
    dist_opp_avg = X_pairs["dist_opp_avg"]
    max_dist_opp_avg = np.max(X_pairs["dist_opp_avg"])
    X_pairs["dist_opp_avg"] = dist_opp_avg / max_dist_opp_avg
    dist_tm_min = X_pairs["dist_tm_min"]
    max_dist_tm_min = np.max(X_pairs["dist_tm_min"])
    X_pairs["dist_tm_min"] = dist_tm_min / max_dist_tm_min
    dist_tm_avg = X_pairs["dist_tm_avg"]
    max_dist_tm_avg = np.max(X_pairs["dist_tm_avg"])
    X_pairs["dist_tm_avg"] = dist_tm_avg / max_dist_tm_avg
    X_pairs = X_pairs.drop(columns=["x_sender","y_sender", "x_j", "y_j"])

    return X_pairs, y_pairs

def same_team_(sender,player_j):
    if sender <= 11:
        return int(player_j <= 11)
    else:
        return int(player_j > 11)

def classic_import_and_save_dataset():

    # Import dataset
    input_dataframe = pd.read_csv('input_training_set.csv', sep=',')
    output_dataframe = pd.read_csv('output_training_set.csv', sep=',')

    x_pairs, y_pairs = classic_make_pair_of_players(input_dataframe, output_dataframe)

    # Save to a csv file
    x_pairs.to_csv('save_x_pairs_classic.csv', header=True, index=True, mode='w')
    y_pairs.to_csv('save_y_pairs_classic.csv', header=True, index=True, mode='w')


def classic_make_pair_of_players(X_, y_=None):
    n_ = X_.shape[0]
    pair_feature_col = ["sender", "x_sender", "y_sender", "player_j", "x_j", "y_j", "same_team",
                        "is_pass_forward", "distance", "dist_opp", "dist_tm"]
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
            dist_opp = avg_dist_opp(sender, player_j, p_i_)
            dist_tm = avg_dist_teammates(sender, player_j, p_i_)
            X_pairs.iloc[idx] = [sender / 22, p_i_["x_{:0.0f}".format(sender)] / MAX_X_ABS_VAL,
                                 p_i_["y_{:0.0f}".format(sender)] / MAX_Y_ABS_VAL,
                                 player_j / 22, p_i_["x_{:0.0f}".format(player_j)] / MAX_X_ABS_VAL,
                                 p_i_["y_{:0.0f}".format(player_j)] / MAX_Y_ABS_VAL,
                                 same_team_(sender, player_j), is_pass_forward(sender, player_j, p_i_), 0, dist_opp,
                                 dist_tm]
            """
            if same_team_(sender, player_j) == 1:
                s_t_dist += distance((p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)]), (p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)]))
                if s_t_dist <= closer_st:
                    closer_st = s_t_dist
            if same_team_(sender, player_j) == 0:
                adv_t_dist += distance((p_i_["x_{:0.0f}".format(sender)], p_i_["y_{:0.0f}".format(sender)]), (p_i_["x_{:0.0f}".format(player_j)], p_i_["y_{:0.0f}".format(player_j)]))
                if adv_t_dist <= closer_adv:
                    closer_adv = adv_t_dist
            """
            if not y_ is None:
                y_pairs.iloc[idx]["pass"] = int(player_j == y_.iloc[i])

            idx += 1
        # idx_end = idx
        # s_t_dist /= 10
        # adv_t_dist /= 11
        # for j in range(idx_start, idx_end):
        # X_pairs.iloc[j]['same_team_moy_dist'] = s_t_dist
        # X_pairs.iloc[j]['adv_team_moy_dist'] = adv_t_dist
        # X_pairs.iloc[j]['closer_same_team_dist'] = closer_st
        # X_pairs.iloc[j]['closer_adv_dist'] = closer_adv
        # X_pairs.iloc[j]['game_zone'] = game_zone
    distance = compute_distance_(X_pairs)
    max_dist = np.max(distance)
    X_pairs["distance"] = distance / max_dist
    dist_opp = X_pairs["dist_opp"]
    max_dist_opp = np.max(X_pairs["dist_opp"])
    X_pairs["dist_opp"] = dist_opp / max_dist_opp
    dist_tm = X_pairs["dist_tm"]
    max_dist_tm = np.max(X_pairs["dist_tm"])
    X_pairs["dist_tm"] = dist_tm / max_dist_tm
    X_pairs = X_pairs.drop(columns=["x_sender", "y_sender", "x_j", "y_j"])

    return X_pairs, y_pairs

def import_and_save_dataset():

    # Import dataset
    input_dataframe = pd.read_csv('input_training_set.csv', sep=',')
    output_dataframe = pd.read_csv('output_training_set.csv', sep=',')

    x_pairs_df, y_pairs_df = make_pair_of_players(input_dataframe, output_dataframe)
    # To numpy
    x_pairs = x_pairs_df.to_numpy()
    y_pairs = y_pairs_df.to_numpy()
    # Count each pass row:
    pass_counter = 0
    for i in range(0, len(y_pairs)):
        if y_pairs[i] == 1:
            pass_counter += 1

    # Create the new arrays:
    new_x = np.zeros((x_pairs.shape[0] + 20 * pass_counter, x_pairs.shape[1]))
    new_y = np.zeros((x_pairs.shape[0] + 20 * pass_counter, y_pairs.shape[1]))

    # copy and multiply by 21 each pass row
    index = 0
    for i in range(0, x_pairs.shape[0]):
        new_x[index] = x_pairs[i]
        new_y[index] = y_pairs[i]
        index += 1
        if y_pairs[i] == 1:
            for j in range(0, 20):
                new_x[index] = x_pairs[i]
                new_y[index] = y_pairs[i]
                index += 1

    x_pairs = pd.DataFrame(data=new_x, columns=x_pairs_df.columns, index=None)
    y_pairs = pd.DataFrame(data=new_y, columns=y_pairs_df.columns, index=None)

    # Save to a csv file
    x_pairs.to_csv('save_x_pairs.csv', header=True, index=True, mode='w')
    y_pairs.to_csv('save_y_pairs.csv', header=True, index=True, mode='w')

def shuffle_dataset(x, y):

    new_x = np.copy(x)
    new_y = np.copy(y)
    # shuffle the dataset:
    indexer = np.arange(new_x.shape[0])
    np.random.shuffle(indexer)
    tmp_x = new_x
    tmp_y = new_y
    new_x = np.zeros(tmp_x.shape)
    new_y = np.zeros(tmp_y.shape)
    for i in range(0, indexer.shape[0]):
        new_x[i] = tmp_x[indexer[i]]
        new_y[i] = tmp_y[indexer[i]]

    return new_x, new_y

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

<<<<<<< HEAD:tf_toy_6.py
    # import_and_save_dataset()
=======
    #import_and_save_dataset()
    #classic_import_and_save_dataset()
    #print('DONE')
>>>>>>> 11b1b8b94bdfab68b22e244aa3399d6dcd6b150d:old/tf_toy_6.py

    # Read dataset:
    x = pd.read_csv('save_x_pairs.csv', sep=',', index_col=0)
    y = pd.read_csv('save_y_pairs.csv', sep=',', index_col=0)
    classic_x = pd.read_csv('save_x_pairs_classic.csv', sep=',', index_col=0)
    classic_y = pd.read_csv('save_y_pairs_classic.csv', sep=',', index_col=0)


    x["same_team"] = (x["same_team"] - 0.5)*2
    x["max_cs"] = (x["max_cs"] + 1) / 2
    x = x.drop(columns=["dist_tm_avg", "dist_tm_min", "dist_opp_min_sender"])

    # Got numpy versions
    x_np = x.to_numpy()
    y_np = y.to_numpy()
    classic_x_np = classic_x.to_numpy()
    classic_y_np = classic_y.to_numpy()

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x_np, y_np, shuffle=False, test_size=0.19995392766)
    c_x_train, c_x_test, c_y_tran, c_y_test = train_test_split(classic_x_np, classic_y_np, shuffle=False, test_size=0.19995392766)
    x_train, y_train = shuffle_dataset(x_train, y_train)
    x_test, y_test = shuffle_dataset(x_test, y_test)
    # Create the model
    model = network()

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
    model.train(x_train, y_train, x_test, y_test)

    pred = model.model.predict(c_x_test)

    print(y_test.shape[0])

    probas = pred.reshape(int(c_y_test.shape[0]/22), 22)

    pred_players = np.argmax(probas, axis=1) + 1

    one_hot_y = c_y_test.reshape(int(y_test.shape[0]/22), 22)

    y_player = np.argmax(one_hot_y, axis=1) + 1

    print(probas[:20])
    print(one_hot_y[:20])
    print(pred_players[:20])
    acc = np.average([1 if e else 0 for e in np.equal(pred_players, y_player)])

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

    print('Test accuracy: {}'.format(acc))
    print('One counter = {} - True one counter = {}'.format(one_counter, true_one_counter))

    temp_x = pd.read_csv('input_training_set.csv', sep=',', index_col="Id")
    print(temp_x.head())
    index = 6948
    x_t = np.zeros(22)
    y_t = np.zeros(22)
    for i in range(1,23):
        x_t[i-1] = temp_x.iloc[index]['x_{:0.0f}'.format(i)]
        y_t[i-1] = temp_x.iloc[index]['y_{:0.0f}'.format(i)]
    no = temp_x.iloc[index, 0]
    print(temp_x.iloc[index]["time_start"])
    plt.plot(x_t[:11], y_t[:11], 'rs', x_t[11:], y_t[11:], 'g^', x_t[no-1], y_t[no-1], 'bo')
    for i in range(1, 23):
        plt.text(x_t[i-1], y_t[i-1],"{}".format(i))
    plt.show()




















