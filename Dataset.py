import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

MAX_X_ABS_VAL = 5250
MAX_Y_ABS_VAL = 3400

class Dataset():

    def __init__(self):

        # Original format training set, Numpy array
        self.original_train_x = None
        self.original_train_y = None

        # Final set:
        self.final_set = None

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

        # Final set in pairs format
        self.final_pairs = None

        # Player-pairs format of the testing set
        self.pairs_test_x = None
        self.pairs_test_y = None

        # Player-pairs format of the validation set
        self.pairs_validation_x = None
        self.pairs_validation_y = None

        # Pairs headers
        self.pairs_x_header = None
        self.pairs_y_header = None

        # Standard scaler:
        self.standard_scaler = StandardScaler()

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
        original_final = pd.DataFrame(self.final_set, columns=self.original_x_header)

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

        # Final set:
        self.final_pairs = self.convertor(original_final, y=None)

    def convertor(self, x, y=None):

        return self.make_players_pairs(x, y)

    def import_original_training(self, split_train=0.9, split_test=0.29, split_val=0.01):

        # Read the csv to pandas
        x_df = pd.read_csv('Original_data/input_training_set.csv', sep=',', index_col=46)
        y_df = pd.read_csv('Original_data/output_training_set.csv', sep=',')
        final_df = pd.read_csv('Original_data/input_test_set.csv', sep=',', index_col=0)

        print(x_df.iloc[3])
        print(final_df.iloc[3])


        # Replace positions axis
        #x_df = self.positions_convertor(x_df)

        # Store headers
        self.original_x_header = x_df.columns
        self.original_y_header = y_df.columns

        # Get np version
        x = x_df.to_numpy()
        y = y_df.to_numpy()
        final = final_df.to_numpy()

        # Split the set
        #x_train, x_t, y_train, y_t = train_test_split(x, y, test_size=(1-split_train), shuffle=True)
        #x_test, x_valid, y_test, y_valid = train_test_split(x_t, y_t, test_size=(split_val / (split_test + split_val)), shuffle=True)
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=split_train, shuffle=True)

        # Store
        self.original_train_x = x_train
        self.original_train_y = y_train
        self.original_test_x = x_test
        self.original_test_y = y_test
        self.final_set = final
        #self.original_validation_x = x_valid
        #self.original_validation_y = y_valid

    def make_players_pairs(self, x_df, y_df):

        # Get columns of dataframes
        x_col = x_df.columns
        if y_df is not None:
            y_col = y_df.columns

        # get numpy version
        x = x_df.to_numpy()
        if y_df is not None:
            y = y_df.to_numpy()
        n = x.shape[0]

        # Get positions sub array:
        x_pos = np.zeros((n, 22))
        y_pos = np.zeros((n, 22))
        for i in range(0, 22):
            x_pos[:, i] = x[:, 2 + i * 2]
            y_pos[:, i] = x[:, 3 + i * 2]

        # Build a Traditional distance matrix 3D:
        dist = np.zeros((n, 22, 22))
        # And a forward distances matrix
        frw_dist = np.zeros((n, 22, 22))
        for i in range(0, 22):
            for j in range(0, 22):
                # Traditional dist
                dist[:, i, j] = np.sqrt(np.power(np.subtract(x_pos[:, i], x_pos[:, j]), 2) +
                                        np.power(np.subtract(y_pos[:, i], y_pos[:, j]), 2))
                # Forward dist
                frw_dist[:, i, j] = np.subtract(x_pos[:, i], x_pos[:, j])


        # Make a matrix to store each pass frame: n passes with 21 potential receiver and 50 features
        n_features = 50
        if y_df is not None:
            n_features += 1
        passes = np.zeros((n, 22, n_features))

        labels = []
        # Index of added columns
        idx = 0
        for i in range(0, n):
            # Set sender
            sender = int(x_df['sender'].iloc[i] - 1)
            passes[i, :, idx] = sender
            # Set receivers:
            rec = np.arange(0, 22)
            passes[i, :, idx+1] = rec
            passes[i, :, idx+2] = x_pos[i, sender]
        idx += 3
        labels.append('sender')
        labels.append('player_j')
        labels.append('sender_x')
        # Get reciever x position
        for i in range(0, 22):
            passes[:, i, idx] = x_pos[:, i]
        idx += 1
        labels.append('player_j_x')
        # Get mask array for same team and not
        t1 = np.zeros(22)
        t2 = np.zeros(22)
        for i in range(0, 22):
            if i < 11:
                t1[i] = 1
            if i >= 11:
                t2[i] = 1

        # Get distances
        for i in range(0, n):
            sender = int(x_df['sender'].iloc[i] - 1)
            # Dist between sender and player_j
            passes[i, :, idx] = dist[i, sender, :]
            # Forward dist between sender and player_j
            passes[i, :, idx+1] = frw_dist[i, sender, :]
            # Same team or not
            if sender < 11:
                passes[i, :, idx+2] = t1
            else:
                passes[i, :, idx+2] = t2
            # Get two closest defender of the sender
            tmp_dst = dist[i, sender, :]
            if sender < 11:
                tmp_defender = tmp_dst[0:11]
                tmp_opp = tmp_dst[11:22]
            else:
                tmp_defender = tmp_dst[11:22]
                tmp_opp = tmp_dst[0:11]
            # Sort:
            tmp_defender = np.sort(tmp_defender)
            tmp_opp = np.sort(tmp_opp)
            passes[i, :, idx+3] = tmp_defender[1]
            passes[i, :, idx+4] = tmp_defender[2]
            passes[i, :, idx+5] = tmp_opp[0]
            passes[i, :, idx+6] = tmp_opp[1]

        idx += 7
        labels.append('pass_dist')
        labels.append('forward_dist')
        labels.append('same_team')
        labels.append('first_sender_defender_dist')
        labels.append('sec_sender_defender_dist')
        labels.append('first_sender_opp_dist')
        labels.append('sec_sender_opp_dist')

        # Compute the two closest defender and opp for reciever
        for i in range(0, n):
            for j in range(0, 11):
                tmp_dst_defender = dist[i, j, 0:11]
                tmp_dst_defender = np.sort(tmp_dst_defender)
                passes[i, j, idx] = tmp_dst_defender[1]
                passes[i, j, idx+1] = tmp_dst_defender[2]
                tmp_dst_opp = dist[i, j, 11:22]
                tmp_dst_opp = np.sort(tmp_dst_opp)
                passes[i, j, idx+2] = tmp_dst_opp[0]
                passes[i, j, idx+3] = tmp_dst_opp[1]
            for j in range(11, 22):
                tmp_dst_defender = dist[i, j, 11:22]
                tmp_dst_defender = np.sort(tmp_dst_defender)
                passes[i, j, idx] = tmp_dst_defender[1]
                passes[i, j, idx+1] = tmp_dst_defender[2]
                tmp_dst_opp = dist[i, j, 0:11]
                tmp_dst_opp = np.sort(tmp_dst_opp)
                passes[i, j, idx+2] = tmp_dst_opp[0]
                passes[i, j, idx+3] = tmp_dst_opp[1]

        labels.append('first_rec_defender_dist')
        labels.append('sec_rec_defender_dist')
        labels.append('first_rec_opp_dist')
        labels.append('sec_rec_opp_dist')
        idx += 4

        # Compute the two shoortest distances between opposed of reciever and trajectory segment
        # Vectorize the function
        vec_seg_dist = np.vectorize(pnt2line, excluded=['xa', 'ya', 'xb', 'yb'], otypes=[float])
        for i in range(0, n):
            #Get sender index:
            sender = int(x_df['sender'].iloc[i]-1)
            # Get sender positions:
            xa = x_pos[i, sender]
            ya = x_pos[i, sender]
            for j in range(0, 22):
                # Get reciever positions
                xb = x_pos[i, j]
                yb = y_pos[i, j]
                # Get other points positions:
                if j < 11:
                    xc = x_pos[i, 11:22]
                    yc = y_pos[i, 11:22]
                    if sender >= 11:
                        xc = np.delete(xc, sender-11)
                        yc = np.delete(yc, sender-11)
                else:
                    xc = x_pos[i, 0:11]
                    yc = y_pos[i, 0:11]
                    if sender < 11:
                        xc = np.delete(xc, sender)
                        yc = np.delete(yc, sender)
                result = vec_seg_dist(xc=np.asarray(xc).ravel(), yc=np.asarray(yc).ravel(), xa=xa, ya=ya, xb=xb, yb=yb)
                passes[i, j, idx] = np.min(result)

        idx += 1
        labels.append('first_opp_dst_line')

        # Mean and std for x position of sender team and sender opposant
        for i in range(0, n):
            # Get the sender index
            sender = int(x_df['sender'][i] - 1)
            # Get others players position
            positions = x_pos[i, :]
            # Store mean and std of the same team
            if sender < 11:
                passes[i, :, idx] = np.mean(positions[0:11])
                passes[i, :, idx+1] = np.std(positions[0:11])
                passes[i, :, idx+2] = np.mean(positions[11:22])
                passes[i, :, idx+3] = np.mean(positions[11:22])
            else:
                passes[i, :, idx] = np.mean(positions[11:22])
                passes[i, :, idx+1] = np.std(positions[11:22])
                passes[i, :, idx+2] = np.mean(positions[0:11])
                passes[i, :, idx+3] = np.mean(positions[0:11])

        idx += 4
        labels.append('sender_team_gravity')
        labels.append('sender_team_dispersion')
        labels.append('sender_opp_gravity')
        labels.append('sender_opp_dispersion')


        # add target column:
        if y_df is not None:
            for i in range(0, n):
                passes[i, y[i]-1, -1] = 1

        # Reshape the matrix in 2D:
        passes = passes.reshape((n*22, n_features))

        # Add missing features headers:
        for i in range(idx, n_features):
            labels.append('feature_{}'.format(i))
        # Convert in dataframe:
        if y_df is not None:
            labels[-1] = 'pass'
        passes_df = pd.DataFrame(passes, columns=labels)

        to_drop = []
        # Extract y output
        if y_df is not None:
            y_opt = passes_df['pass'].to_numpy()
            y_opt = pd.DataFrame(y_opt, columns=['pass'])
            to_drop.append('pass')
        # Drop some columns
        for names in labels:
            if 'feature_' in names:
                to_drop.append(names)
        to_drop.append('sender')
        to_drop.append('player_j')
        passes_df.drop(columns=to_drop, inplace=True)

        # DATA NORMALIZATION:
        passes_df['pass_dist'] /= MAX_Y_ABS_VAL
        passes_df['forward_dist'] /= MAX_Y_ABS_VAL
        passes_df['first_sender_defender_dist'] /= MAX_Y_ABS_VAL
        passes_df['sec_sender_defender_dist'] /= MAX_Y_ABS_VAL
        passes_df['first_sender_opp_dist'] /= MAX_Y_ABS_VAL
        passes_df['sec_sender_opp_dist'] /= MAX_Y_ABS_VAL
        passes_df['first_rec_defender_dist'] /= MAX_Y_ABS_VAL
        passes_df['sec_rec_defender_dist'] /= MAX_Y_ABS_VAL
        passes_df['first_rec_opp_dist'] /= MAX_Y_ABS_VAL
        passes_df['sec_rec_opp_dist'] /= MAX_Y_ABS_VAL
        passes_df['first_opp_dst_line'] /= MAX_Y_ABS_VAL
        passes_df['sender_team_gravity'] /= MAX_Y_ABS_VAL
        passes_df['sender_team_dispersion'] /= MAX_Y_ABS_VAL
        passes_df['sender_opp_gravity'] /= MAX_Y_ABS_VAL
        passes_df['sender_opp_dispersion'] /= MAX_Y_ABS_VAL
        passes_df['sender_x'] /= MAX_Y_ABS_VAL
        passes_df['player_j_x'] /= MAX_Y_ABS_VAL

        if y_df is not None:
            return passes_df, y_opt
        else:
            return passes_df
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

         final_set = pd.DataFrame(self.final_set, columns=self.original_x_header)
         final_set.to_csv('personal_data/final_set.csv', sep=',', header=True, index=True)
         final_pairs = pd.DataFrame(self.final_pairs, columns=self.pairs_x_header)
         final_pairs.to_csv('personal_data/final_pairs.csv', sep=',', header=True, index=True)

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
        final_set = pd.read_csv('personal_data/final_set.csv', sep=',', index_col=0)
        self.final_set = final_set.to_numpy()
        final_pairs = pd.read_csv('personal_data/final_pairs.csv', sep=',', index_col=0)
        self.final_pairs = final_pairs.to_numpy()







def dot(v,w):
    x,y = v
    X,Y = w
    return x*X + y*Y

def length(v):
    x,y = v
    return math.sqrt(x*x + y*y)

def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)

def unit(v):
    x,y = v
    mag = length(v)
    return (x/mag, y/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y = v
    return (x * sc, y * sc)

def add(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)

def pnt2line(xc, yc, xa, ya, xb, yb):
    """
    Distance between a point and a line segment
    Adapted from the code find on StackOverflow website
    Source:
    https://stackoverflow.com/questions/27161533/find-the-shortest-distance-between-a-point-and-line-segments-not-line
    """
    if (xc == xa and yc == ya) or (xc == xb and yc == yb) or (xa == xb and ya == yb):
        return 999999999
    pnt=(xc, yc)
    start=(xa, ya)
    end=(xb, yb)
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return dist