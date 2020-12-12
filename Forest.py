import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import Dataset
import pickle

class Forest():

    def __init__(self):

        # Dataset
        self.dataset = None

        # Number of trees
        self.forest_size = 100

        # Init the forest
        self.rf = RandomForestRegressor(n_estimators=200, warm_start=False)

    def set_dataset(self, dataset):
        self.dataset = dataset

    def train(self):
        """
        # Adapt the training set to have as pass as no pass
        LS_x = np.zeros((self.dataset.pairs_train_x.shape[0] + int(self.dataset.pairs_train_x.shape[0] * 20 / 22),
                         self.dataset.pairs_train_x.shape[1]))
        LS_y = np.zeros((LS_x.shape[0], 1))

        idx_a = 0
        idx_b = 0
        idx_c = 0
        for i in range(0, self.dataset.pairs_train_x.shape[0]):
            if self.dataset.pairs_train_y[i] == 1:
                LS_x[idx_a:idx_b, :] = self.dataset.pairs_train_x[idx_c:i, :]
                LS_y[idx_a:idx_b, :] = self.dataset.pairs_train_y[idx_c:i, :]
                for j in range(0, 21):
                    LS_x[idx_b+j, :] = self.dataset.pairs_train_x[i, :]
                    LS_y[idx_b+j, :] = self.dataset.pairs_train_y[i, :]

                idx_c = i + 1
                idx_b += 21
                idx_a = idx_b
            else:
                idx_b += 1
        """
        LS_x = self.dataset.pairs_train_x
        LS_y = self.dataset.pairs_train_y

        # Fit the forest

        self.rf.fit(LS_x, LS_y.ravel())

        # Make predictions:
        TS_x = self.dataset.pairs_test_x
        TS_y = self.dataset.pairs_test_y

        pred = self.rf.predict(TS_x)

        # Compute the error:
        error = abs(pred - TS_y)
        print('Testing error {}'.format(np.mean(error)))



    def save_model(self, file_name='saved_models/forest_1.pkl'):

        # Serialize the model
        pickle.dump(self.rf, open(file_name, 'wb'))

    def restore_model(self, file_name='saved_models/forest_1.pkl'):

        # Un-serialize
        self.rf = pickle.load(open(file_name, 'rb'))

