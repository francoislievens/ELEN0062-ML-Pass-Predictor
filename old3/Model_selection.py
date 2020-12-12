import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import Dataset

class Model_selection():

    def __init__(self):

        # ================================================ #
        #           Parameters values to test:             #
        # ================================================ #

        # Number of hidden layers:
        self.nb_HL_range = np.arange(3, 15)
        # Size of hidden layers:
        self.size_HL_range = [40, 60, 80, 100, 120, 140, 160]
        # Activation to test:
        self.activation_range = ['tanh', 'relu']

        # Make an array of each combinations to test:
        self.params_array = []
        for i in range(0, len(self.nb_HL_range)):
            for j in range(0, len(self.size_HL_range)):
                for k in range(0, len(self.activation_range)):
                    self.params_array.append([self.nb_HL_range[i], self.size_HL_range[j], self.activation_range[k]])

        # Import a dataset to use:
        self.dataset = Dataset.Dataset()
        # Restire if from a pre-compiled dataset csv
        self.dataset.restore_dataset()

        # Get the number of features:
        self.nb_active_feat = len(self.dataset.pairs_x_header)


    def selector(self):

        it = 0

        for i in range(0, len(self.params_array)):
            print('* =========================================================================================== ')
            print('* Model selector: ')
            print('* Iteration: {}'.format(it))
            print('* Number of Hidden layers: {}'.format(self.params_array[i][0]))
            print('* Size of the hidden layers {}'.format(self.params_array[i][1]))
            print('* Activation function of hidden layers {}'.format(self.params_array[i][2]))
            print('* =========================================================================================== ')


            # Fill the parameters dictionary:
            options = {
                'nb_HL': self.params_array[i][0],
                'nb_feat': self.nb_active_feat,
                'HL_size': self.params_array[i][1],
                'HL_activ': self.params_array[i][2]
            }