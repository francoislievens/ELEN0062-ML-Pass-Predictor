
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
        #               Default parameters:                #
        # ================================================ #

        # Number of hidden layers:
        self.nb_HL = 5
        # Number of features in the input data
        self.nb_feat = 60
        # Size of hidden layers
        self.HL_size = 60
        # Activation function type
        self.HL_activ = 'tanh'

        # Default Layers list:
        self.layers_lst = [
            ('tanh', 18),
            ('relu', 18),
            ('tanh', 18),
            ('relu', 15)
        ]


        if options != None:
            self.nb_HL = options['nb_HL']
            self.nb_feat = options['nb_feat']
            self.HL_size = options['HL_size']
            self.HL_activ = options['HL_activ']

        # Create the model:
        self.model = tf.keras.models.Sequential()
        # Use float 64
        tf.keras.backend.set_floatx('float64')

        # ================================================ #
        #                    Layers:                       #
        # ================================================ #

        lyr = self.layers_lst
        if options is not None and 'layers' in options.keys():
            lyr = options['layers']

        # The array to store layers:
        self.model_layers = []
        for tpl in lyr:
            self.model.add(tf.keras.layers.Dense(tpl[1], activation=tpl[0]))
        # Add the output layers
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # ================================================ #
        #                  Evaluation:                     #
        # ================================================ #

        # Loss computer:
        self.train_loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.test_loss_object = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.train_sparse_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.test_sparse_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # Accuracy computer:
        self.train_accuracy_object = tf.keras.metrics.Accuracy()
        self.test_accuracy_object = tf.keras.metrics.Accuracy()

        # Loss accumulator:
        self.train_loss_store = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss_store = tf.keras.metrics.Mean(name='test_loss')
        self.train_loss_sparse_store = tf.keras.metrics.Mean(name='train_loss_sparse')
        self.test_loss_sparse_store = tf.keras.metrics.Mean(name='test_loss_sparse')
        # Accuracy accumulator:
        self.train_accu_store = tf.keras.metrics.Mean(name='train_accuracy')
        self.test_accu_store = tf.keras.metrics.Mean(name='test_accuracy')

        # Optimizer
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        # Training_set to use
        self.dataset = None

    @tf.function
    def train_step(self, x_train, y_train, x_test, y_test, s_weights_train, s_weights_test,
                   original_y_train, original_y_test):
        """
        Training function
        """
        # ================================================ #
        #                  Training part                   #
        # ================================================ #
        with tf.GradientTape() as tape:     # To capture errors for the gradient modification
            # Make prediction
            train_predictions = self.model(x_train)
            # Get the error:
            train_loss = self.train_loss_object(y_train, train_predictions, sample_weight=s_weights_train)
        # Compute the gradient who respect the loss
        gradients = tape.gradient(train_loss, self.model.trainable_variables)
        # Change weights of the model
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # Store losses:
        self.train_loss_store(train_loss)

        # ================================================ #
        #          Perf Tracking: training set             #
        # ================================================ #
        # Compute Train accuracy:
        train_predictions = tf.reshape(train_predictions, shape=[-1, 22])
        # Make a softmax to normalize probabilities
        train_predictions = tf.nn.softmax(train_predictions, axis=1)
        # Get loss
        train_loss = self.train_sparse_loss_object(original_y_train, train_predictions)
        self.train_loss_sparse_store(train_loss)

        # And accuracy:
        final_prd = tf.argmax(train_predictions, axis=1)
        self.train_accuracy_object.reset_states()
        self.train_accuracy_object(final_prd, original_y_train)
        self.train_accu_store(self.train_accuracy_object.result())

        # ================================================ #
        #          Perf Tracking: testing set              #
        # ================================================ #

        # Make predictions on testing set
        pred = self.model(x_test)
        # First classical loss:
        test_loss = self.test_loss_object(y_test, pred, sample_weight=s_weights_test)
        self.test_loss_store(test_loss)
        # Receiver prediction loss
        pred = tf.reshape(pred, shape=[-1, 22])
        pred = tf.nn.softmax(pred, axis=1)
        loss = self.test_sparse_loss_object(original_y_test, pred)
        self.test_loss_sparse_store(loss)

        # And accuracy:
        final_pred = tf.math.argmax(pred, axis=1)
        self.test_accuracy_object.reset_states()
        self.test_accuracy_object(final_pred, original_y_test)
        self.test_accu_store(self.test_accuracy_object.result())


    def train(self, report=False):
        nb_epoch = 200

        x_train = self.dataset.pairs_train_x
        y_train = self.dataset.pairs_train_y
        x_test = self.dataset.pairs_test_x
        y_test = self.dataset.pairs_test_y
        original_y_train = np.copy(self.dataset.original_train_y)
        original_y_test = np.copy(self.dataset.original_test_y)
        original_y_train -= 1
        original_y_test -= 1

        # Store performances
        tracker = np.zeros((nb_epoch, 7))

        # Build sample weights:
        s_weights_train = np.ones(x_train.shape[0])
        s_weights_test = np.ones(x_test.shape[0])
        for i in range(0, x_train.shape[0]):
            if y_train[i] == 1:
                s_weights_train[i] = 21
        for i in range(0, x_test.shape[0]):
            if y_test[i] == 1:
                s_weights_test[i] = 21

        s_weights_test = y_test * 21
        for epoch in range(0, nb_epoch):
            for _ in range(0, 20):
                # Make a train step
                self.train_step(x_train, y_train, x_test, y_test, s_weights_train, s_weights_test, original_y_train,
                                original_y_test)

            print('------------------------')
            print('Epoch: {}'.format(epoch * 20))
            # Print the loss: return the mean of all error in the accumulator
            print('Test Loss : %s' % self.test_loss_store.result())
            print('Train Loss: %s' % self.train_loss_store.result())
            print('Test Loss player pred: %s' % self.test_loss_sparse_store.result())
            print('Train Loss player pred: %s' % self.train_loss_sparse_store.result())
            print('Test Accuracy: %s' % self.test_accu_store.result())
            print('Train Accuracy: %s' % self.train_accu_store.result())

            # Store results:
            tracker[epoch, 0] = epoch * 20
            tracker[epoch, 1] = self.test_loss_store.result()
            tracker[epoch, 2] = self.train_loss_store.result()
            tracker[epoch, 3] = self.test_loss_sparse_store.result()
            tracker[epoch, 4] = self.train_loss_sparse_store.result()
            tracker[epoch, 5] = self.test_accu_store.result()
            tracker[epoch, 6] = self.train_accu_store.result()

            # Reset the accumulator
            self.train_loss_store.reset_states()
            self.test_loss_store.reset_states()
            self.train_accu_store.reset_states()
            self.train_accu_store.reset_states()
            self.test_loss_sparse_store.reset_states()
            self.train_loss_sparse_store.reset_states()

        if report:
            return tracker

    def set_dataset(self, dataset):

        self.dataset = dataset












