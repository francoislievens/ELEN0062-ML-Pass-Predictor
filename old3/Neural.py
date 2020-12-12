
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import Dataset


class Neural():

    def __init__(self, options=None):

        # ================================================ #
        #            Model Hyper-parameters                #
        # ================================================ #

        # Number of hidden layers:
        self.nb_HL = 6
        # Number of features in the input data
        self.nb_feat = 60
        # Size of hidden layers
        self.HL_size = 60
        # Activation function type
        self.HL_activ = 'relu'
        # Batch size for the learning set WARNING: use multiples of 22
        self.batch_size = 22*1000
        # Number of epoch to perform:
        self.nb_epoch = 200

        if options != None:
            self.nb_HL = options['nb_HL']
            self.nb_feat = options['nb_feat']
            self.HL_size = options['HL_size']
            self.HL_activ = options['HL_activ']

        # ================================================ #
        #                Create the model                  #
        # ================================================ #
        self.model = tf.keras.models.Sequential()

        # Add input layers
        self.model.add(tf.keras.layers.Dense(self.nb_feat, activation=self.HL_activ))
        # Add Hidden layers
        self.hidden_layers = []
        for i in range(0, self.nb_HL):
            self.hidden_layers.append(self.model.add(tf.keras.layers.Dense(self.HL_size, activation=self.HL_activ)))
        # Add output layer
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Primary Loss computer
        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        # Secondary Loss computer : to compute receiver predictions performances
        self.secondary_loss_object = tf.keras.losses.CategoricalCrossentropy()

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # Loss tracker
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_loss_secondary = tf.keras.metrics.Mean(name='test_loss_secondary')

        # Accuracy tracker
        self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        self.test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')
        self.test_accuracy_secondary = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy_secondary')

        # Training_set to use
        self.dataset = None



    @tf.function
    def train_step(self, x_p_train, y_p_train, y_p_train_weights):
        """
        One step of learning
        :param x_p_train: Input in pairs of players format
        :param y_p_train: Target of pairs of players format: 1 if pass, 0 if not
        :param y_p_train_weights: Weights to accord to each output: 1 have *21 weights
        """
        with tf.GradientTape() as tape:
            # Make predictions
            predictions = self.model(x_p_train)
            # Get the loss
            loss = self.loss_object(y_p_train, predictions, sample_weight=y_p_train_weights)
        # Determine gradients from the loss
        grd = tape.gradient(loss, self.model.trainable_variables)
        # Change weights
        self.optimizer.apply_gradients(zip(grd, self.model.trainable_variables))
        # Store performances
        self.train_loss(loss)
        self.train_accuracy(y_p_train, predictions, sample_weight=y_p_train_weights)

    @tf.function
    def test_step(self, x_p_test, y_p_test, y_p_test_weights, y_original):
        """
        Test predictions of the neural network on a testing set. Compute test
        loss and accuracy on the pairs of players dataset and accuracy on receiver
        prediction
        :param x_p_test: Test inputs in pairs of players format
        :param y_p_test: Targets in pairs of players format
        :param y_p_test_weights: weights accorded to pairs of players output
        :param y_original: Targets in normal format: the number of the receiver of the pass
        WARNING: index in y_original (nb of each player) must start from ZERO
        """
        # Make predictions on the test set:
        predictions = self.model(x_p_test)
        # Compute the normal pair of players loss and accuracy:
        loss = self.loss_object(y_p_test, predictions, sample_weight=y_p_test_weights)
        self.test_loss(loss)
        self.test_accuracy(y_p_test, predictions, sample_weight=y_p_test_weights)
        # Compute the prediction of the receiver:
        predictions_b = tf.reshape(tensor=predictions, shape=[-1, 22])
        y_b = tf.reshape(tensor=y_p_test, shape=[-1, 22])
        y_b = tf.cast(y_b, dtype=tf.float32)
        loss_2 = self.secondary_loss_object(y_b, predictions_b)
        self.test_loss_secondary(loss_2)
        self.test_accuracy_secondary(y_b, predictions_b)

    def train(self, sets=(0, 0), nb_epoch=20, history=False):

        LS_x = self.dataset.pairs_train_x
        LS_y = self.dataset.pairs_train_y
        TS_x = self.dataset.pairs_test_x
        TS_y = self.dataset.pairs_test_y
        TS_orig_y = self.dataset.original_test_y

        if sets is not None:
            if len(sets) == 5:
                LS_x, LS_y, TS_x, TS_y, TS_orig_y = sets

        # Adapt indexes of players from zero
        TS_orig_y -= 1

        # Compute sample weights
        LS_y_weights = np.copy(LS_y)
        LS_y_weights *= 21
        TS_y_weights = np.copy(TS_y)
        TS_y_weights *= 21

        # Store performances:
        train_loss_ar = []
        train_acc_ar = []
        test_loss_ar = []
        test_acc_ar = []
        test_sec_loss_ar = []
        test_sec_acc_ar = []

        # ================================================ #
        #                  Training Loop                   #
        # ================================================ #

        for epoch in range(0, nb_epoch):

            # Training step:

            # Make training step:
            self.train_step(LS_x, LS_y, LS_y_weights)
            # Track performances:
            print('Training: Loss: {}, Accuracy: {}'.format(self.train_loss.result(),
                                                            self.train_accuracy.result()), end="")

            # Testing step:
            self.test_step(TS_x, TS_y, TS_y_weights, TS_orig_y)
            # Track performances
            print('\nEpoch {}, Pairs Loss: {}, pairs Accuracy: {},'
                  'Original Loss: {}, Original Accuracy: {}'.format(epoch+1, self.test_loss.result(),
                                                                    self.test_accuracy.result(),
                                                                    self.test_loss_secondary.result(),
                                                                    self.test_accuracy_secondary.result()))

            # Store in vectors:
            if history:
                train_loss_ar.append(self.train_loss.result())
                train_acc_ar.append(self.train_accuracy.result())
                test_loss_ar.append(self.test_loss.result())
                test_acc_ar.append(self.test_accuracy.result())
                test_sec_loss_ar.append(self.test_loss_secondary.result())
                test_sec_acc_ar.append(self.test_accuracy_secondary.result())
            # Reset accumulators:
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.test_loss_secondary.reset_states()
            self.test_accuracy_secondary.reset_states()

        if history:
            return [train_loss_ar, train_acc_ar, test_loss_ar, test_acc_ar, test_sec_loss_ar, test_sec_acc_ar]

    def set_dataset(self, dataset):
        self.dataset = dataset
