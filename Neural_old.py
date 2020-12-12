
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

        if options != None:
            self.nb_HL = options['nb_HL']
            self.nb_feat = options['nb_feat']
            self.HL_size = options['HL_size']
            self.HL_activ = options['HL_activ']

        # Create the model:
        self.model = tf.keras.models.Sequential()
        # Use float 64
        #tf.keras.backend.set_floatx('float64')
        # Add layers

        #self.model.add(tf.keras.layers.Dense(self.nb_feat, activation=self.HL_activ))
        #self.hidden_layers = []
        #for i in range(0, self.nb_HL):
        #    self.hidden_layers.append(self.model.add(tf.keras.layers.Dense(self.HL_size, activation=self.HL_activ)))

        #self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        self.model.add(tf.keras.layers.Dense(50, activation='relu'))
        self.model.add(tf.keras.layers.Dense(50, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(50, activation='relu'))
        self.model.add(tf.keras.layers.Dense(25, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(15, activation='relu'))
        self.model.add(tf.keras.layers.Dense(10, activation='tanh'))
        #self.model.add(tf.keras.layers.Dropout(0.2))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Loss computer:
        self.train_loss_object = tf.keras.losses.BinaryCrossentropy()
        self.test_loss_object = tf.keras.losses.BinaryCrossentropy()
        # Accuracy Accumulator:
        self.test_acc = tf.keras.metrics.Accuracy()

        # Loss accumulator:
        self.train_loss_accu = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss_accu = tf.keras.metrics.Mean(name='Test_loss')
        # Accuraccy accumulator:


        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # Training_set to use
        self.dataset = None

    @tf.function
    def train_step(self, x_train, y_train, x_test, y_test, s_weights_train, s_weights_test, original_y):
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
            # Compute test accuracy:
            test_predictions = self.model(x_test)
            test_predictions = tf.reshape(test_predictions, shape=[-1, 22])
            final_pred = tf.math.argmax(test_predictions, axis=1)
            target = tf.cast(original_y, dtype=tf.int64)
            target = tf.reshape(target, [-1])
            self.test_acc.reset_states()
            self.test_acc.update_state(final_pred, target)
            self.test_loss_accu(self.test_acc.result())
            #counter = tf.math.equal(final_pred, target)
            #test_loss = tf.reduce_sum(tf.cast(counter, tf.float32))
            #self.test_loss_accu(test_loss / tf.size(counter, out_type=tf.float32))


    def train(self):

        x_train = self.dataset.pairs_train_x
        y_train = self.dataset.pairs_train_y
        x_test = self.dataset.pairs_test_x
        y_test = self.dataset.pairs_test_y
        original_y = self.dataset.original_test_y
        original_y -= 1

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
        for epoch in range(0, 32):
            for _ in range(0, 50):
                # Make a train step
                self.train_step(x_train, y_train, x_test, y_test, s_weights_train, s_weights_test, original_y)

            print('Epoch: {}'.format(epoch))
            # Print the loss: return the mean of all error in the accumulator
            print('Test Loss: %s' % self.test_loss_accu.result())
            print('Train Loss: %s' % self.train_loss_accu.result())
            # Reset the accumulator
            self.train_loss_accu.reset_states()
            self.test_loss_accu.reset_states()

            # Last test:
            prd = self.model(x_test).numpy()
            prd = np.reshape(prd, (-1, 22))
            prd_final = np.argmax(prd, axis=1)
            result = np.zeros(len(prd_final))
            for i in range(0, prd.shape[0]):
                #print('{} - {}'.format(prd_final[i], original_y[i]))
                if prd_final[i] == original_y[i]:
                    result[i] = 1

            print('Final accuracy: {}'.format(np.mean(result)))


    def train_features_selections(self, x_train, y_train, x_test, y_test, original_y):

        original_y -= 1

        # Build sample weights:
        s_weights_train = np.ones(x_train.shape[0])
        s_weights_test = np.ones(x_test.shape[0])
        for i in range(0, x_train.shape[0]):
            if y_train[i] == 1:
                s_weights_train[i] = 21
        for i in range(0, x_test.shape[0]):
            if y_test[i] == 1:
                s_weights_test[i] = 21

        train_curve = []
        test_curve = []

        s_weights_test = y_test * 21
        for epoch in range(0, 150):
            for _ in range(0, 50):
                # Make a train step
                self.train_step(x_train, y_train, x_test, y_test, s_weights_train, s_weights_test, original_y)
            # Get losses
            test_loss_tmp = self.test_loss_accu.result()
            train_loss_tmp = self.train_loss_accu.result()
            # Store losses
            train_curve.append(train_loss_tmp.numpy())
            test_curve.append(test_loss_tmp.numpy())

            print('Epoch: {}'.format(epoch))
            # Print the loss: return the mean of all error in the accumulator
            print('Test Loss: %s' % test_loss_tmp)
            print('Train Loss: %s' % train_loss_tmp)
            # Reset the accumulator
            self.train_loss_accu.reset_states()
            self.test_loss_accu.reset_states()

            if epoch >= 2:
                print('=======BREAK=======')
                return train_curve, test_curve



    def predict_pass(self, df_x):

        # Adapt inputs according to features of the learning set
        x_pairs = self.dataset.convertor(df_x)
        # Numpy version
        df_np = df_x.to_numpy()

        print(self.dataset.pairs_train_x.shape)
        print(x_pairs.shape)

        # Make predictions:
        pred = self.model(x_pairs.to_numpy())
        # Reshape:
        pred = tf.reshape(pred, [-1, 22])
        # Store final numeric prediction:
        final_pred = tf.math.argmax(pred, axis=1)

        return final_pred

    def true_loss(self, pred, target):

        pred = np.reshape(pred, (-1, 22))
        final_pred = np.argmax(pred, axis=1)
        return np.sum(final_pred == target) / len(final_pred)

    def error_pass(self, predict, data):
        """
        Compute the % of good pass predictions
        :param predict: An array of integers who contains recievers predictions
        :param data: Real recievers
        :return: the probability to make good predictions
        """

        return np.sum(predict == data) / len(predict)


    def set_dataset(self, dataset):

        self.dataset = dataset












