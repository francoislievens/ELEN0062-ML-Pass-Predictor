
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import dataset_tuner


class Neural():

    def __init__(self):
        # Create the model:
        self.model = tf.keras.models.Sequential()

        # Add layers
        self.model.add(tf.keras.layers.Dense(7, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(7, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(7, activation='tanh'))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Loss computer:
        self.train_loss_object = tf.keras.losses.MeanAbsoluteError()
        self.test_loss_object = tf.keras.losses.MeanAbsoluteError()


        # Loss accumulator:
        self.train_loss_accu = tf.keras.metrics.Mean(name='Train_loss')
        self.test_loss_accu = tf.keras.metrics.Mean(name='Test_loss')

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam()

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
        # Compute the gradient who respect the loss
        gradients = tape.gradient(train_loss, self.model.trainable_variables)
        # Change weights of the model
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # Compute test error:
        test_predictions = self.model(x_test)
        test_loss = self.train_loss_object(y_test, test_predictions)
        # Store losses:
        self.train_loss_accu(train_loss)
        self.test_loss_accu(test_loss)

    def train(self, x_train, y_train, x_test, y_test):

        for epoch in range(0, 20):
            for _ in range(0, 100):
                # Make a train step
                self.train_step(x_train, y_train, x_test, y_test)

            print('Epoch: {}'.format(epoch))
            # Print the loss: return the mean of all error in the accumulator
            print('Test Loss: %s' % self.test_loss_accu.result())
            print('Train Loss: %s' % self.train_loss_accu.result())
            # Reset the accumulator
            self.train_loss_accu.reset_states()
            self.test_loss_accu.reset_states()

    def predict_pass(self, df_x):

        # Switch to pairs of players
        x_pairs, y_pairs = dataset_tuner.make_pair_of_players(df_x)
        print(x_pairs.shape)
        # Go to numpy array
        np_x_pairs = x_pairs.to_numpy()

        # Make predictions
        predictions = self.model.predict(np_x_pairs)

        # Store result:
        result = np.zeros(df_x.shape[0], dtype=np.int)
        result_best = np.zeros(df_x.shape[0])
        idx = 0

        for i in range(0, int(predictions.shape[0]/22)):
            for j in range(0, 22):
                if predictions[idx] >= result_best[i]:
                    result_best[i] = predictions[idx]
                    result[i] = int(np_x_pairs[idx][1] * 22)
                idx += 1
        return result

def build_training_set(proportion=0.8):

    # Inport raw dataset
    x = pd.read_csv('input_training_set.csv', sep=',')
    y = pd.read_csv('output_training_set.csv', sep=',')

    # Keep wanted proportion
    end_index = int(x.shape[0] * proportion)
    x_tmp = np.copy(x.to_numpy()[0:end_index, :])
    y_tmp = np.copy(y.to_numpy()[0:end_index, :])

    # As dataframe:
    x_tmp = pd.DataFrame(data=x_tmp, columns=x.columns)
    y_tmp = pd.DataFrame(data=y_tmp, columns=y.columns)

    # Make pairs of players
    x_pairs, y_pairs = dataset_tuner.make_pair_of_players(x_tmp, y_tmp)

    # Balance the dataset between pass and no pass
    x_pairs, y_pairs = dataset_tuner.ballance_dataset(x_pairs, y_pairs)

    # Shuffle the dataset
    x_pairs, y_pairs = dataset_tuner.shuffle_dataset(x_pairs, y_pairs)

    # Write the dataset in a file
    x_pairs.to_csv('personal_set/training_x_pairs.csv', sep=',')
    y_pairs.to_csv('personal_set/training_y_pairs.csv', sep=',')



if __name__ == '__main__':

    # Build training set
    # build_training_set(0.8)

    # Build the model:
    model = Neural()

    # Import prepared training set
    x_train_pairs = pd.read_csv('personal_set/training_x_pairs.csv', index_col=0, sep=',')
    y_train_pairs = pd.read_csv('personal_set/training_y_pairs.csv', index_col=0, sep=',')

    # Split the set:
    x_train, x_test, y_train, y_test = train_test_split(x_train_pairs, y_train_pairs, test_size=0.2)

    # Train the model:
    model.train(x_train.to_numpy(), y_train.to_numpy(), x_test.to_numpy(), y_test.to_numpy())

    # Import raw data never seen
    raw_x = pd.read_csv('input_training_set.csv', sep=',')
    raw_y = pd.read_csv('output_training_set.csv', sep=',')

    # keep only the 20% unused:
    start_index = int(raw_x.shape[0] * 0.8)
    tmp_x = np.copy(raw_x.to_numpy())[start_index:, :]
    tmp_y = np.copy(raw_y.to_numpy())[start_index:, :]
    raw_x = pd.DataFrame(data=tmp_x, columns=raw_x.columns)
    raw_y = pd.DataFrame(data=tmp_y, columns=raw_y.columns)

    # Make a prediction
    predictions = model.predict_pass(raw_x)

    # Compare with output

    true_class = 0
    for i in range(0, len(predictions)):
        print('{} - {}'.format(predictions[i], tmp_y[i]))
        if predictions[i] == tmp_y[i][0]:
            true_class += 1

    true_class /= tmp_y.shape[0]

    print('proportion bien classée: {}'.format(true_class))








