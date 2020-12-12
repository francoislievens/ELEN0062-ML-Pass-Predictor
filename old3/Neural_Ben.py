
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import Dataset


class Neural():

    def __init__(self):
        # Create the model:
        self.model = tf.keras.models.Sequential()

        # Add layers
        self.model.add(tf.keras.layers.Dense(60, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(60, activation='relu'))
        self.model.add(tf.keras.layers.Dense(16, activation='tanh'))
        self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Loss computer:
        self.train_loss_object = tf.keras.losses.BinaryCrossentropy()
        self.test_loss_object = tf.keras.losses.BinaryCrossentropy()

        # Loss accumulator:
        self.train_loss_accu = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss_accu = tf.keras.metrics.Mean(name='Test_loss')

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # Training_set to use
        self.dataset = None

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
        # Compute the gradient who respect the loss
        gradients = tape.gradient(train_loss, self.model.trainable_variables)
        # Change weights of the model
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # Store losses:
        self.train_loss_accu(train_loss)
        # If test error wanted

        if x_test is not None:
            # Compute test error:
            test_predictions = self.model(x_test)
            test_loss = self.train_loss_object(y_test, test_predictions, sample_weight=s_weights_test)
            self.test_loss_accu(test_loss)


    def train(self):

        x_train = self.dataset.pairs_train_x
        y_train = self.dataset.pairs_train_y
        x_test = self.dataset.pairs_test_x
        y_test = self.dataset.pairs_test_y

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
        for epoch in range(0, 20):
            for _ in range(0, 100):
                # Make a train step
                self.train_step(x_train, y_train, x_test, y_test, s_weights_train, s_weights_test)

            print('Epoch: {}'.format(epoch))
            # Print the loss: return the mean of all error in the accumulator
            print('Test Loss: %s' % self.test_loss_accu.result())
            print('Train Loss: %s' % self.train_loss_accu.result())
            # Reset the accumulator
            self.train_loss_accu.reset_states()
            self.test_loss_accu.reset_states()

    def predict_pass(self, df_x):

        # Adapt inputs according to features of the learning set
        x_pairs = self.dataset.convertor(df_x)
        # Numpy version
        df_np = df_x.to_numpy()

        # Make predictions:
        pred = self.model.predict(x_pairs)
        # Reshape:
        pred = np.reshape(pred, (-1, 22))
        # Store final numeric prediction:
        final_pred = np.zeros(pred.shape[0])
        for i in range(0, pred.shape[0]):
            final_pred[i] = np.argmax(pred[i, :]) + 1

        return final_pred

    def set_dataset(self, dataset):

        self.dataset = dataset












