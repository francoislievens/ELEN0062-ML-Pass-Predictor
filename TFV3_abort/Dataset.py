import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import Features_computers

MAX_X_ABS_VAL = 5250
MAX_Y_ABS_VAL = 3400

class Dataset():

    def __init__(self):

        # Original format training set, Numpy array
        self.original_train_x = None
        self.original_train_y = None

        # Original format testing set, Numpy array + headers array
        self.original_test_x = None
        self.original_test_y = None

        # Original format validation set, Numpy array + headers array
        self.original_validation_x = None
        self.original_validation_y = None

        # Original set headers
        self.original_x_header = None
        self.original_y_header = None

        # Player-pairs format of the training set
        self.pairs_train_x = None
        self.pairs_train_y = None

        # Player-pairs format of the testing set
        self.pairs_test_x = None
        self.pairs_test_y = None

        # Player-pairs format of the validation set
        self.pairs_validation_x = None
        self.pairs_validation_y = None

        # Pairs headers
        self.pairs_x_header = None
        self.pairs_y_header = None

    def import_original_training(self, split_train=0.7, split_test=0.2, split_val=0.1):

        # Read the csv to pandas
        x_df = pd.read_csv('Original_data/input_training_set.csv', sep=',')
        y_df = pd.read_csv('Original_data/output_training_set.csv', sep=',')

        # Store headers
        self.original_x_header = x_df.columns
        self.original_y_header = y_df.columns

        # Get np version
        x = x_df.to_numpy()
        y = y_df.to_numpy()

        # Split the set
        x_train, x_t, y_train, y_t = train_test_split(x, y, test_size=(1-split_train), shuffle=False)
        x_test, x_valid, y_test, y_valid = train_test_split(x_t, y_t, test_size=(split_val / (split_test + split_val)), shuffle=False)

        # Store
        self.original_train_x = x_train
        self.original_train_y = y_train
        self.original_test_x = x_test
        self.original_test_y = y_test
        self.original_validation_x = x_valid
        self.original_validation_y = y_valid

    def convertor(self, x, y=None):
        """
        Adapt a standard dataset to the pairs form with new features
        """
        # Compute distance matrix:
        dist_matrix = Features_computers.players_distances(x)
        # Construct pairs dataset:
        y_pairs = None
        if y is not None:
            x_pairs, y_pairs = self.make_players_pairs(x, y)
        else:
            x_pairs = self.make_players_pairs(x)

        # Add features:

        # Is pass forward:
        x_pairs = Features_computers.is_pass_forward(x_pairs)
        # Pass distance:
        x_pairs = Features_computers.pass_distance(x_pairs)
        # Compute min, avg and std distance between same team and opposant, sender and reciever:
        x_pairs = Features_computers.dist_tool(x_pairs, dist_matrix)
        # Compute gravity centers of each team
        x_pairs = Features_computers.gravity_center(x_pairs, x)
        # Normalize the dataset
        x_pairs = Features_computers.normalizer(x_pairs)
        # Drop pass index column
        x_pairs.drop(columns=['pass_index'], inplace=True)
        # Delete all empty columns:
        headers = x_pairs.columns
        to_drop = []
        for name in headers:
            if 'feature_' in name:
                to_drop.append(name)
        x_pairs.drop(columns=to_drop, inplace=True)





        if y is not None:
            return x_pairs, y_pairs
        else:
            return x_pairs

    def learning_set_builders(self):
        """
        This method adapt the imported original learning set in pairs of
        players form with new features
        """
        # Transform in a dataframe:
        original_train_x = pd.DataFrame(self.original_train_x, columns=self.original_x_header)
        original_train_y = pd.DataFrame(self.original_train_y, columns=self.original_y_header)
        original_test_x = pd.DataFrame(self.original_test_x, columns=self.original_x_header)
        original_test_y = pd.DataFrame(self.original_test_y, columns=self.original_y_header)
        original_validation_x = pd.DataFrame(self.original_validation_x, columns=self.original_x_header)
        original_validation_y = pd.DataFrame(self.original_validation_y, columns=self.original_y_header)

        # Training set
        x, y = self.convertor(original_train_x, original_train_y)
        self.pairs_train_x = x.to_numpy()
        self.pairs_train_y = y.to_numpy()

        # Headers
        self.pairs_x_header = x.columns
        self.pairs_y_header = y.columns

        # Testing set
        x, y = self.convertor(original_test_x, original_test_y)
        self.pairs_test_x = x.to_numpy()
        self.pairs_test_y = y.to_numpy()

        # Validation set
        x, y = self.convertor(original_validation_x, original_validation_y)
        self.pairs_validation_x = x.to_numpy()
        self.pairs_validation_y = y.to_numpy()




    def make_players_pairs(self, x, y=None):
        """
        Take an original format dataframe as input and return
        a player_pairs format dataframe in array who accept
        50 features
        """
        # Get numpy version:
        x_np = x.to_numpy()
        y_np = None
        if y is not None:
            y_np = y.to_numpy()
        # Get the number of entries:
        n = x.shape[0]
        # Get position sub_array:
        p_x_pos = np.zeros((n, 22))
        p_y_pos = np.zeros((n, 22))
        for i in range(0, 22):
            p_x_pos[:, i] = x_np[:, 2 + i * 2]
            p_y_pos[:, i] = x_np[:, 3 + i * 2]
        # Make a matrix to store each pass frame: n passes with 21 potential receiver and 50 features
        n_features = 50
        if y is not None:
            n_features += 1
        passes = np.zeros((n, 22, n_features))
        # Copy sender
        for i in range(0, n):
            sender = x['sender'].iloc[i]
            passes[i, :, 0:3] += [sender, x['x_{:0.0f}'.format(sender)].iloc[i], x['y_{:0.0f}'.format(sender)].iloc[i]]
            # The index of the pass
            passes[i, :, 7] = i
        # Copy receivers
        rc = np.arange(1, 23, dtype=float)
        passes[:, :, 3] = rc
        # Position for each receiver
        for i in range(0, n):
            passes[i, :, 4] += p_x_pos[i, :]
            passes[i, :, 5] += p_y_pos[i, :]
        # Same team (1) or not (0)
        for i in range(0, n):
            same_team = 1
            if passes[i][0][0] > 11:
                same_team = 0
            passes[i, 0:12, 6] = same_team
            passes[i, 12:23, 6] = 1 - same_team

        # add target column:
        if y is not None:
            for i in range(0, n):
                passes[i, y_np[i]-1, -1] = 1

        # Reshape the matrix in 2D:
        passes = passes.reshape((n*22, n_features))

        # Y output:
        y_opt = None
        if y is not None:
            y_opt = passes[:, n_features-1]
            y_opt = pd.DataFrame(y_opt, columns=['pass'])

        # X output
        x_opt = passes[:, :-1]
        # Get dataframe headers:
        x_header = ['sender', 'x_sender', 'y_sender', 'player_j', 'x_j', 'y_j', 'same_team', 'pass_index']
        for i in range(8, n_features - 1):
            x_header.append('feature_{}'.format(i))
        # In dataframe
        x_opt = pd.DataFrame(x_opt, columns=x_header)

        if y is not None:
            return x_opt, y_opt
        else:
            return x_opt

    def save_dataset(self):

         original_train_x = pd.DataFrame(self.original_train_x, columns=self.original_x_header)
         original_train_x.to_csv('personal_data/original_train_x.csv', sep=',', header=True, index=True)
         original_train_y = pd.DataFrame(self.original_train_y, columns=self.original_y_header)
         original_train_y.to_csv('personal_data/original_train_y.csv', sep=',', header=True, index=True)
         original_test_x = pd.DataFrame(self.original_test_x, columns=self.original_x_header)
         original_test_x.to_csv('personal_data/original_test_x.csv', sep=',', header=True, index=True)
         original_test_y = pd.DataFrame(self.original_test_y, columns=self.original_y_header)
         original_test_y.to_csv('personal_data/original_test_y.csv', sep=',', header=True, index=True)
         original_valid_x = pd.DataFrame(self.original_validation_x, columns=self.original_x_header)
         original_valid_x.to_csv('personal_data/original_valid_x.csv', sep=',', header=True, index=True)
         original_valid_y = pd.DataFrame(self.original_validation_y, columns=self.original_y_header)
         original_valid_y.to_csv('personal_data/original_valid_y.csv', sep=',', header=True, index=True)

         pairs_train_x = pd.DataFrame(self.pairs_train_x, columns=self.pairs_x_header)
         pairs_train_x.to_csv('personal_data/pairs_train_x.csv', sep=',', header=True, index=True)
         pairs_train_y = pd.DataFrame(self.pairs_train_y, columns=self.pairs_y_header)
         pairs_train_y.to_csv('personal_data/pairs_train_y.csv', sep=',', header=True, index=True)
         pairs_test_x = pd.DataFrame(self.pairs_test_x, columns=self.pairs_x_header)
         pairs_test_x.to_csv('personal_data/pairs_test_x.csv', sep=',', header=True, index=True)
         pairs_test_y = pd.DataFrame(self.pairs_test_y, columns=self.pairs_y_header)
         pairs_test_y.to_csv('personal_data/pairs_test_y.csv', sep=',', header=True, index=True)
         pairs_valid_x = pd.DataFrame(self.pairs_validation_x, columns=self.pairs_x_header)
         pairs_valid_x.to_csv('personal_data/pairs_valid_x.csv', sep=',', header=True, index=True)
         pairs_valid_y = pd.DataFrame(self.pairs_validation_y, columns=self.pairs_y_header)
         pairs_valid_y.to_csv('personal_data/pairs_valid_y.csv', sep=',', header=True, index=True)

    def restore_dataset(self):

        original_train_x = pd.read_csv('personal_data/original_train_x.csv', sep=',', index_col=0)
        self.original_x_header = original_train_x.columns
        self.original_train_x = original_train_x.to_numpy()
        original_train_y = pd.read_csv('personal_data/original_train_y.csv', sep=',', index_col=0)
        self.original_y_header = original_train_y.columns
        self.original_train_y = original_train_y.to_numpy()
        original_test_x = pd.read_csv('personal_data/original_test_x.csv', sep=',', index_col=0)
        self.original_test_x = original_test_x.to_numpy()
        original_test_y = pd.read_csv('personal_data/original_test_y.csv', sep=',', index_col=0)
        self.original_test_y = original_test_y.to_numpy()
        original_valid_x = pd.read_csv('personal_data/original_valid_x.csv', sep=',', index_col=0)
        self.original_validation_x = original_valid_x.to_numpy()
        original_valid_y = pd.read_csv('personal_data/original_valid_y.csv', sep=',', index_col=0)
        self.original_validation_y = original_valid_y.to_numpy()

        pairs_train_x = pd.read_csv('personal_data/pairs_train_x.csv', sep=',', index_col=0)
        self.pairs_train_x = pairs_train_x.to_numpy()
        self.pairs_x_header = pairs_train_x.columns
        pairs_train_y = pd.read_csv('personal_data/pairs_train_y.csv', sep=',', index_col=0)
        self.pairs_train_y = pairs_train_y.to_numpy()
        self.pairs_y_header = pairs_train_y.columns
        pairs_test_x = pd.read_csv('personal_data/pairs_test_x.csv', sep=',', index_col=0)
        self.pairs_test_x = pairs_test_x.to_numpy()
        pairs_test_y = pd.read_csv('personal_data/pairs_test_y.csv', sep=',', index_col=0)
        self.pairs_test_y = pairs_test_y.to_numpy()
        pairs_valid_x = pd.read_csv('personal_data/pairs_valid_x.csv', sep=',', index_col=0)
        self.pairs_validation_x = pairs_valid_x.to_numpy()
        pairs_valid_y = pd.read_csv('personal_data/pairs_valid_y.csv', sep=',', index_col=0)
        self.pairs_validation_y = pairs_valid_y.to_numpy()







