import pandas as pd
import numpy as np

MAX_X_ABS_VAL = 5250
MAX_Y_ABS_VAL = 3400

def make_pair_of_players(X_, y_=None):
    n_ = X_.shape[0]
    pair_feature_col = ["sender", "x_sender", "y_sender", "player_j", "x_j", "y_j", "same_team",
                        "is_pass_forward", "distance", "dist_opp", "dist_tm"]
    X_pairs = pd.DataFrame(data=np.zeros((n_ * 22, len(pair_feature_col))), columns=pair_feature_col)
    y_pairs = pd.DataFrame(data=np.zeros((n_ * 22, 1)), columns=["pass"])

    # From pass to pair of players
    idx = 0
    for i in range(n_):
        sender = X_.iloc[i].sender
        players = np.arange(1, 23)
        # other_players = np.delete(players, sender-1)
        p_i_ = X_.iloc[i]
        # Mean dist from same team
        s_t_dist = 0
        # Mean dist adv
        adv_t_dist = 0
        idx_start = idx
        # Closer same team dist
        closer_st = 5000
        # Closer adv dist
        closer_adv = 5000
        # Game zone
        game_zone = 0
        if X_.iloc[i]["x_{:0.0f}".format(sender)] < 0:
            game_zone = 1
        for player_j in players:
            dist_opp = avg_dist_opp(sender, player_j, p_i_)
            dist_tm = avg_dist_teammates(sender, player_j, p_i_)
            X_pairs.iloc[idx] = [sender / 22, p_i_["x_{:0.0f}".format(sender)] / MAX_X_ABS_VAL,
                                 p_i_["y_{:0.0f}".format(sender)] / MAX_Y_ABS_VAL,
                                 player_j / 22, p_i_["x_{:0.0f}".format(player_j)] / MAX_X_ABS_VAL,
                                 p_i_["y_{:0.0f}".format(player_j)] / MAX_Y_ABS_VAL,
                                 same_team_(sender, player_j), is_pass_forward(sender, player_j, p_i_), 0, dist_opp,
                                 dist_tm]

            if not y_ is None:
                y_pairs.iloc[idx]["pass"] = int(player_j == y_.iloc[i])

            idx += 1

    distance = compute_distance_(X_pairs)
    max_dist = np.max(distance)
    X_pairs["distance"] = distance / max_dist
    dist_opp = X_pairs["dist_opp"]
    max_dist_opp = np.max(X_pairs["dist_opp"])
    X_pairs["dist_opp"] = dist_opp / max_dist_opp
    dist_tm = X_pairs["dist_tm"]
    max_dist_tm = np.max(X_pairs["dist_tm"])
    X_pairs["dist_tm"] = dist_tm / max_dist_tm
    X_pairs = X_pairs.drop(columns=["x_sender", "y_sender", "x_j", "y_j"])

    return X_pairs, y_pairs

def is_pass_forward(sender, receiver, columns):
    sender_x = columns["x_{:0.0f}".format(sender)]
    receiver_x = columns["x_{:0.0f}".format(receiver)]
    left_most = MAX_X_ABS_VAL
    left_most_player = 0
    for player in range(1, 23):
        player_x = columns["x_{:0.0f}".format(player)]
        if left_most > player_x:
            left_most = player_x
            left_most_player = player
    same_team = 0
    if left_most_player < 12:
        if sender < 12:
            same_team = 1
    else:
        if sender >= 12:
            same_team = 1
    if same_team == 1:
        return np.abs(sender_x - receiver_x)/MAX_X_ABS_VAL if (receiver_x - left_most) > (sender_x - left_most) else -np.abs(sender_x - receiver_x)/MAX_X_ABS_VAL
    else:
        return np.abs(sender_x - receiver_x)/MAX_X_ABS_VAL if (receiver_x - left_most) < (sender_x - left_most) else -np.abs(sender_x - receiver_x)/MAX_X_ABS_VAL

def get_diagonale():
    return np.sqrt(MAX_X_ABS_VAL**2 + MAX_Y_ABS_VAL**2)

def compute_distance_(X_):
    d = np.zeros((X_.shape[0],))
    d = np.sqrt((X_["x_sender"]-X_["x_j"])**2 + (X_["y_sender"]-X_["y_j"])**2)
    return d

def avg_dist_teammates(sender, receiver, columns):
    if sender == receiver:
        return 0
    sender_x = columns["x_{:0.0f}".format(sender)]
    sender_y = columns["y_{:0.0f}".format(sender)]
    receiver_x = columns["x_{:0.0f}".format(receiver)]
    receiver_y = columns["y_{:0.0f}".format(receiver)]
    dist_sender_receiver = np.sqrt((sender_x - receiver_x)**2 + (sender_y - receiver_y)**2)
    dist = get_diagonale()
    for player in range(1,23):
        if player == sender or player == receiver or same_team_(player, sender) != 1:
            continue
        player_x = columns["x_{:0.0f}".format(player)]
        player_y = columns["y_{:0.0f}".format(player)]
        dist_player_sender = np.sqrt((sender_x - player_x)**2 + (sender_y - player_y)**2)
        dist_player_receiver = np.sqrt((player_x - receiver_x)**2 + (player_y - receiver_y)**2)
        dist = min(dist, dist_player_receiver + dist_player_sender)
    return dist

def avg_dist_opp(sender, receiver, columns):
    if sender == receiver:
        return 0
    sender_x = columns["x_{:0.0f}".format(sender)]
    sender_y = columns["y_{:0.0f}".format(sender)]
    receiver_x = columns["x_{:0.0f}".format(receiver)]
    receiver_y = columns["y_{:0.0f}".format(receiver)]
    dist_sender_receiver = np.sqrt((sender_x - receiver_x)**2 + (sender_y - receiver_y)**2)
    dist = get_diagonale()
    for player in range(1,23):
        if player == sender or player == receiver or same_team_(player, sender) == 1:
            continue
        player_x = columns["x_{:0.0f}".format(player)]
        player_y = columns["y_{:0.0f}".format(player)]
        dist_player_sender = np.sqrt((sender_x - player_x)**2 + (sender_y - player_y)**2)
        dist_player_receiver = np.sqrt((player_x - receiver_x)**2 + (player_y - receiver_y)**2)
        dist = min(dist, dist_player_receiver + dist_player_sender)
    return dist

def same_team_(sender,player_j):
    if sender <= 11:
        return int(player_j <= 11)
    else:
        return int(player_j > 11)

def ballance_dataset(x_pairs, y_pairs):

    # Get numpy version
    np_x = x_pairs.to_numpy()
    np_y = y_pairs.to_numpy()

    # Count number of pass and get the new_dataset length:
    nb_pass = np.sum(np_y)
    new_size = int(np_x.shape[0] - nb_pass + nb_pass * 21)

    # Init new dataset
    new_x = np.zeros((new_size, np_x.shape[1]))
    new_y = np.zeros((new_size, np_y.shape[1]))

    # Copy and multiply lines
    idx = 0
    for i in range(0, np_x.shape[0]):
        # Copy line
        new_x[idx] = np_x[i]
        new_y[idx] = np_y[i]
        idx += 1
        if np_y[i] == 1:
            for i in range(0, 20):
                new_x[idx] = np_x[i]
                new_y[idx] = np_y[i]
                idx += 1
    # Get dataframe format
    df_x = pd.DataFrame(data=new_x, columns=x_pairs.columns, index=None)
    df_y = pd.DataFrame(data=new_y, columns=y_pairs.columns, index=None)
    return df_x, df_y

def shuffle_dataset(x, y):

    # Get numpy copy
    new_x = np.copy(x.to_numpy())
    new_y = np.copy(y.to_numpy())

    # shuffle the dataset:
    indexer = np.arange(new_x.shape[0])
    np.random.shuffle(indexer)
    tmp_x = new_x
    tmp_y = new_y
    new_x = np.zeros(tmp_x.shape)
    new_y = np.zeros(tmp_y.shape)
    for i in range(0, indexer.shape[0]):
        new_x[i] = tmp_x[indexer[i]]
        new_y[i] = tmp_y[indexer[i]]

    # Back on dataset form
    df_x = pd.DataFrame(data=new_x, columns=x.columns, index=None)
    df_y = pd.DataFrame(data=new_y, columns=y.columns, index=None)
    return df_x, df_y