import numpy as np
import pandas as pd
from Neural import Neural
from Dataset import Dataset

class Model_select():

    def __init__(self):

        # Create the dataset
        self.dataset = Dataset()
        # Import the dataset
        #self.dataset.import_original_training(split_train=0.8)
        # Compute pairs of players form
        #self.dataset.learning_set_builders()

        self.dataset.restore_dataset()

        # Number of layers to test
        self.nb_layers = []
        for i in range(2, 10):
            self.nb_layers.append(i)

        # Size of layers
        self.size_layers = []
        idx = 19
        for i in range(0, 10):
            self.size_layers.append(idx + i*3)

        # Activation function
        self.activator = ['tanh', 'relu', 'mixt']

        # Learning rate:
        self.learning_rate = [0.1, 0.01, 0.001, 0.0001]


        # Create a global array with combinations to test. In case of crash, we can so start again at a given index
        self.global_array = []
        for number in self.nb_layers:
            for width in self.size_layers:
                for activ in self.activator:
                    for lr in self.learning_rate:

                        self.global_array.append((number, width, activ, lr))

        # Get an array with already tested combinations:
        self.to_do = np.arange(0, len(self.global_array))

        # Headers of output files
        self.headers = ['idx', 'best_accu', 'activation', 'nb_layers', 'layers_width', 'learning_rate' ]
        for i in range(0, 30):
            self.headers.append('{}'.format((i+1)*20))

    def start(self):

        itt = 0
        while len(self.to_do > 0):
            itt += 1

            # Find a random inex
            idx = int(np.random.choice(self.to_do, 1))
            # Update to do
            self.to_do = np.delete(self.to_do, np.where(self.to_do == idx))

            # Create parameters of the model:
            layers_lst = []
            # Get parameters:
            activation = self.global_array[idx][2]
            nbr = self.global_array[idx][0]
            width = self.global_array[idx][1]
            lr = self.global_array[idx][3]

            print('========================================================')
            print(' Model selection: iter {} '.format(itt))
            print(' Remain to do: {} / {}'.format(len(self.to_do), len(self.global_array)))
            print(' Activation = {}'.format(activation))
            print(' Number of layers = {}'.format(nbr))
            print(' Neurones per layers = {}'.format(width))
            print(' Learning rate = {}'.format(lr))
            print('========================================================')

            # List each layers
            for i in range(0, nbr):
                layers_lst.append([activation, width])
            if activation == 'mixt':
                for i in range(0, len(layers_lst)):
                    if i % 2 == 0:
                        layers_lst[i][0] = 'tanh'
                    else:
                        layers_lst[i][0] = 'relu'


            options = {
                'layers_lst': layers_lst,
                'learning_rate': lr
            }
            # Build the model
            model = Neural(options=options)
            # Import the dataset
            model.set_dataset(self.dataset)

            # Train the model on 50 x 20 epochs
            results = model.train(report=True, nb_epoch=30, silent=True)

            # Get the best accuracy:
            best_acc = np.max(results[:, 5])

            # Get string array to write in files array
            str_global = [str(idx), str(best_acc), activation, str(nbr), str(width), str(lr)]
            str_test_loss = str_global.copy()
            str_train_loss = str_global.copy()
            str_test_loss_sparse = str_global.copy()
            str_train_loss_sparse = str_global.copy()
            str_test_accu = str_global.copy()
            str_train_accu = str_global.copy()

            for i in range(0, results.shape[0]):
                str_test_loss.append(str(results[i, 1]))
                str_train_loss.append(str(results[i, 2]))
                str_test_loss_sparse.append(str(results[i, 3]))
                str_train_loss_sparse.append(str(results[i, 4]))
                str_test_accu.append(str(results[i, 5]))
                str_train_accu.append(str(results[i, 6]))

            # open files
            f_test_loss = open('model_selection/train_loss.csv', 'a')
            f_train_loss = open('model_selection/testloss.csv', 'a')
            f_test_loss_sparse = open('model_selection/train_loss_sparse.csv', 'a')
            f_train_loss_sparse = open('model_selection/train_loss_sparse.csv', 'a')
            f_test_accu = open('model_selection/test_accu.csv', 'a')
            f_train_accu = open('model_selection/train_accu.csv', 'a')

            # Write files
            f_test_loss.write(','.join(str_test_loss))
            f_train_loss.write(','.join(str_train_loss))
            f_test_loss_sparse.write(','.join(str_test_loss_sparse))
            f_train_loss_sparse.write(','.join(str_train_loss_sparse))
            f_test_accu.write(','.join(str_test_accu))
            f_train_accu.write(','.join(str_train_accu))
            f_test_loss.write('\n')
            f_train_loss.write('\n')
            f_test_loss_sparse.write('\n')
            f_train_loss_sparse.write('\n')
            f_test_accu.write('\n')
            f_train_accu.write('\n')

            # Close files
            f_test_loss.close()
            f_train_loss.close()
            f_test_loss_sparse.close()
            f_train_loss_sparse.close()
            f_test_accu.close()
            f_train_accu.close()

    def warm_start(self):

        df = pd.read_csv('model_selection/train_loss.csv', sep=',')
        df = pd.DataFrame(df.to_numpy(), columns=self.headers)
        print(df)

        idx_col = df['idx'].to_numpy()

        for item in idx_col:
            self.to_do = np.delete(self.to_do, np.where(self.to_do == item))


if __name__ == '__main__':

    optimizer = Model_select()

    optimizer.warm_start()

    optimizer.start()