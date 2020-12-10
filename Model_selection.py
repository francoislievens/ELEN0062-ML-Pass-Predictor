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
        self.nb_HL_range = np.arange(1, 15)
        # Size of hidden layers:
        self.size_HL_range = [40, 60, 80, 100, 120, 140, 160]