import numpy as np
import pandas as pd
import Dataset
import Neural

class Features_selection():

    def __init__(self):

        # Init the model:
        self.model = Neural.Neural()
        # Get the dataset:
        self.dataset = Dataset.Dataset()
        # Restore the dataset from pre_processing
        self.dataset.restore_dataset()
        # Set the dataset
        self.model.set_dataset(self.dataset)

        # Deleted Features:
        self.nb_active_feat = self.dataset.pairs_test_x.shape[1]
        self.del_feat = np.zeros(self.nb_active_feat, dtype=bool)


    def reset(self):

        self.model = Neural.Neural()
        self.model.set_dataset(self.dataset)

    def selector(self):

        it = 0

        while self.nb_active_feat > 5:

            # Store best accuracy for this round of remove test
            rmv_results_acc = []
            rmv_results_idx = []

            # Try to remove each of already present features
            #for i in range(0, self.del_feat.shape[0]):
            for i in range(0, 2):

                print('* =========================================================================================== ')
                print('* Features selector: ')
                print('* Iteration: {}'.format(it))
                print('* Number of active features: {}'.format(self.nb_active_feat))
                print('* =========================================================================================== ')

                # Only if already present
                if not self.del_feat[i]:

                    # Get parameters for the new network:
                    options = {
                        'nb_HL': 6,
                        'nb_feat': self.nb_active_feat - 1,
                        'HL_size': self.nb_active_feat - 1,
                        'HL_activ': 'tanh'
                    }

                    # Reset the network
                    self.model = Neural.Neural(options)
                    self.model.set_dataset(self.dataset)

                    # Compute create the training and testing set:
                    LS = np.zeros((self.dataset.pairs_train_x.shape[0], self.nb_active_feat - 1))
                    TS = np.zeros((self.dataset.pairs_test_x.shape[0], self.nb_active_feat - 1))
                    idx = 0
                    for j in range(0, self.del_feat.shape[0]):
                        # Don't use the feature i
                        if j != i and not self.del_feat[j]:
                            LS[:, idx] = self.dataset.pairs_train_x[:, j]
                            TS[:, idx] = self.dataset.pairs_test_x[:, j]
                            idx += 1

                    train_curve, test_curve = self.model.train_features_selections(LS, self.dataset.pairs_train_y,
                                                                                   TS, self.dataset.pairs_test_y,
                                                                                   self.dataset.original_test_y)

                    # Store the best accuracy:
                    best_accu = np.max(test_curve)
                    rmv_results_acc.append(best_accu)
                    rmv_results_idx.append(i)

                    # Make a string array to write result in a file:
                    test_str_arr = []
                    train_str_arr = []
                    # Write deleted features
                    for j in range(0, len(self.del_feat)):
                        if j == i:
                            test_str_arr.append(str(True))
                            train_str_arr.append(str(True))
                        else:
                            test_str_arr.append(str(self.del_feat[j]))
                            train_str_arr.append(str(self.del_feat[j]))
                    # write best accuracy
                    test_str_arr.append(str(best_accu))
                    train_str_arr.append(str(np.min(train_curve)))
                    # Write accuracy evolution
                    for epoch in range(0, 200):
                        if epoch < len(test_curve):
                            test_str_arr.append(str(test_curve[epoch]))
                            train_str_arr.append(str(train_curve))
                        else:
                            test_str_arr.append('nan')
                            train_str_arr.append('nan')
                    # Open the file in append mode
                    file_test = open('model_selection/features_sel_test.csv', 'a')
                    file_train = open('model_selection/features_sel_train.csv', 'a')
                    # Write
                    file_test.write(';'.join(test_str_arr))
                    file_test.write('\n')
                    file_test.close()
                    file_train.write(';'.join(train_str_arr))
                    file_train.write('\n')
                    file_train.close()

                    it += 1


            # Delete the feature who have the worst impact on the accuracy:
            worst = (1, 0)
            for i in range(0, len(rmv_results_acc)):
                if rmv_results_acc[i] <= worst[0]:
                    worst = (rmv_results_acc[i], rmv_results_idx[i])

            # Close the feature:
            self.del_feat[worst[1]] = True
            self.nb_active_feat -= 1


if __name__ == '__main__':

    FS = Features_selection()
    FS.selector()





