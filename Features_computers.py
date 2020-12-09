import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pw
from functools import reduce

MAX_X_ABS_VAL = 5250
MAX_Y_ABS_VAL = 3400

def get_diagonale():
    return np.sqrt(MAX_X_ABS_VAL**2 + MAX_Y_ABS_VAL**2)

def is_pass_forward(pairs):

    # check the first empty feature:
    col = pairs.columns
    idx = 0
    for i in range(0, len(col)):
        if 'feature_' in col[i]:
            idx = i
            break
    # Get numpy version
    np_pairs = pairs.to_numpy()
    # Get the mask of forward pass
    mask = (np_pairs[:, 1] <= np_pairs[:, 4])
    mask = np.asarray(mask, dtype=float)
    np_pairs[:, idx] = mask

    # Update header
    new_col = []
    for i in range(0, len(col)):
        if i == idx:
            new_col.append('is_pass_forward')
        else:
            new_col.append(col[i])

    return pd.DataFrame(np_pairs, columns=new_col)

def get_row_index(row):
    return row[int(row[-1])]

def set_row_index(row, val):
    row[int(row[-1])] = val


def my_cos_sim(arr):
    x, y = arr[:2], arr[2:]
    return pw.cosine_similarity(np.array([x]), np.array([y]))

def max_cosine_similarity(pairs, x, y):
    np_pairs = pairs.to_numpy()
    sender = np_pairs[:, 0].astype(int)
    print(sender)
    rec = np_pairs[:, 3].astype(int)
    sender_x = np.apply_along_axis( get_row_index, 1, np.hstack((x, sender.reshape((sender.shape[0], 1))-1)))
    sender_y = np.apply_along_axis( get_row_index, 1, np.hstack((y, sender.reshape((sender.shape[0], 1))-1)))
    rec_x = np.apply_along_axis( get_row_index, 1, np.hstack((x, rec.reshape((rec.shape[0], 1))-1)))
    rec_y = np.apply_along_axis( get_row_index, 1, np.hstack((y, rec.reshape((rec.shape[0], 1))-1)))
    receiver_vector = np.zeros((sender_x.shape[0], 2))
    receiver_vector[:, 0] = sender_x - rec_x
    receiver_vector[:, 1] = sender_y - rec_y
    cos = np.zeros(np_pairs.shape[0]) - 1
    for player in range(0,22):
        # trouver tous les indices où cos_sim doit être à -1 
        player_x = x[:, player]
        player_y = y[:, player]
        opponent_vector = np.zeros((sender_x.shape[0], 2))
        opponent_vector[:, 0] = sender_x - player_x
        opponent_vector[:, 1] = sender_y - player_y
        # condition = (sender == player or sender == receiver or (sender > 12 and player > 12) or (sender <= 11 and player <= 11))
        invalid_sender = np.where(sender-1 == player)
        invalid_rec = np.where(rec-1 == player)
        if player > 12:
            same_team = np.where(sender > 12)    
        else:
            same_team = np.where(sender <= 11)
        print('time 1')
        not_valid = reduce(np.union1d, (invalid_sender, invalid_rec, same_team))
        print('time 2')
        cos_sim = np.apply_along_axis(my_cos_sim, 1, np.hstack((receiver_vector, opponent_vector)))
        print('time 3')
        # remplacer par -1/0 pour chacun de ces indices dans cos_sim 
        np.put(cos_sim, not_valid, np.zeros(not_valid.shape[0])-1)
        print(cos_sim)
        pairs["max_cos_sim_{}".format(player+1)] = np.concatenate(cos_sim).ravel()
    return pairs
"""
def get_dist_from_adv_goal(pairs):
    np_pairs = pairs.to_numpy()
    left_most = np.ones(np_pairs.shape[0]) * MAX_X_ABS_VAL
    left_most_player = np.zeros(np_pairs.shape[0])
    for player in range(0, 22):
        player_x = np_pairs[:, 4]
        left_most = np.minimum(left_most, player_x)
        left_most_player = player
    same_team = 0
    left_most_player += 1
    receivers = np_pairs[:, 3]
    team1 = np.zeros(np_pairs.shape[0]) - 5250
    team2 = np.zeros(np_pairs.shape[0]) + 5250
    goal = np.where((receivers < 12 and left_most_player < 12) or (receivers >= 12 and left_most_player >= 12), team2, team1)
    # Compute distance
    distances = np.sqrt(np.power(goal - np_pairs[:, 4], 2) +
                               np.power(np_pairs[:, 5], 2)) / get_diagonale()
    print(distances)
    pairs['distance_goal'] = distances
    return pairs
"""

def get_grid_feature(pairs):
    np_pairs = pairs.to_numpy()
    x_sender = np_pairs[:, 1]
    y_sender = np_pairs[:, 2]
    x_rec = np_pairs[:, 4]
    y_rec = np_pairs[:, 5]
    sender_position = get_grid_position(x_sender, y_sender)
    rec_position = get_grid_position(x_rec, y_rec)
    pairs["rec_grid_pos"] = rec_position
    return pairs

def get_grid_position(x, y):
    nb_col = 3
    nb_row = 3
    row = nb_row * (x + MAX_X_ABS_VAL) / (2 * MAX_X_ABS_VAL)
    col = nb_col * (y + MAX_Y_ABS_VAL) / (2 * MAX_Y_ABS_VAL)
    row = np.round(row-0.5) - 1
    col = np.round(col-0.5) - 1
    position = np.add((row + 1) * nb_col, col)
    return position / (nb_col * nb_row)



def pass_distance(pairs):
    # Get numpy version
    np_pairs = pairs.to_numpy()

    # Compute distance
    distances = np.sqrt(np.power(np_pairs[:, 1] - np_pairs[:, 4], 2) +
                               np.power(np_pairs[:, 2] - np_pairs[:, 5], 2)) / get_diagonale()
    print(distances)
    pairs['distance'] = distances
    return pairs

def dist_tool(pairs, dist):
    """
    Compute distance between sender and receiver + dist between each opposant and the sender +
    dist between the same opposant and the receiver. The shoortest dist is chosen
    Compute Mean and min
    """
    # check the first empty feature:
    col = pairs.columns
    idx = 0
    for i in range(0, len(col)):
        if 'feature_' in col[i]:
            idx = i
            break
    # Get numpy version
    np_pairs = pairs.to_numpy()
    # initial for the where condition
    init = np.zeros(11)
    init *= 5250

    for i in range(0, pairs.shape[0]):
        sender = int(np_pairs[i, 0] - 1)
        rec = int(np_pairs[i, 3] - 1)
        pass_idx = int(np_pairs[i, 7])
        # Get distances
        sender_dists= dist[pass_idx, sender, :]
        rec_dists = dist[pass_idx, rec, :]
        team = 0
        if rec >= 11:
            team = 1

        tmp_opp_dist_sum1 = []
        tmp_team_dist_sum1 = []
        tmp_opp_dist_sum2 = []
        tmp_team_dist_sum2 = []
        tmp_opp_dist_sum3 = []
        tmp_team_dist_sum3 = []
        for player in range(0, 22):
            if player != sender and player != rec:
                # Same team:
                if player < 11 and team == 0 or player >= 11 and team == 1:
                    tmp_team_dist_sum1.append(sender_dists[player] + rec_dists[player])
                    tmp_team_dist_sum2.append(sender_dists[player])
                    tmp_team_dist_sum3.append(rec_dists[player])
                # Opposite team:
                if player < 11 and team == 1 or player >= 11 and team == 0:
                    tmp_opp_dist_sum1.append(sender_dists[player] + rec_dists[player])
                    tmp_opp_dist_sum2.append(sender_dists[player])
                    tmp_opp_dist_sum3.append(rec_dists[player])
        """
        #for each player
        for j in range(0, 22):
            for k in range(0, 22):
                if j != sender and j != rec and k != sender and k != rec:
                    # Same team:
                    if j < 11 and team == 0 and k < 11 or j >= 11 and team == 1 and k >= 11:
                        tmp_team_dist_sum1.append(sender_dists[j] + rec_dists[k])
                    # Opposite team:
                    if j < 11 and team == 1 and k < 11 or j >= 11 and team == 0 and k >= 11:
                        tmp_opp_dist_sum1.append(sender_dists[j] + rec_dists[k])
        """


        np_pairs[i, idx] = np.min(tmp_opp_dist_sum1)
        np_pairs[i, idx+1] = np.mean(tmp_opp_dist_sum1)
        np_pairs[i, idx+2] = np.std(tmp_opp_dist_sum1)
        np_pairs[i, idx+3] = np.min(tmp_team_dist_sum1)
        np_pairs[i, idx+4] = np.mean(tmp_team_dist_sum1)
        np_pairs[i, idx+5] = np.std(tmp_team_dist_sum1)

        np_pairs[i, idx+6] = np.min(tmp_opp_dist_sum2)
        np_pairs[i, idx+7] = np.mean(tmp_opp_dist_sum2)
        np_pairs[i, idx+8] = np.std(tmp_opp_dist_sum2)
        np_pairs[i, idx+9] = np.min(tmp_team_dist_sum2)
        np_pairs[i, idx+10] = np.mean(tmp_team_dist_sum2)
        np_pairs[i, idx+11] = np.std(tmp_team_dist_sum2)

        np_pairs[i, idx+12] = np.min(tmp_opp_dist_sum3)
        np_pairs[i, idx+13] = np.mean(tmp_opp_dist_sum3)
        np_pairs[i, idx+14] = np.std(tmp_opp_dist_sum3)
        np_pairs[i, idx+15] = np.min(tmp_team_dist_sum3)
        np_pairs[i, idx+16] = np.mean(tmp_team_dist_sum3)
        np_pairs[i, idx+17] = np.std(tmp_team_dist_sum3)

    # Update headers
    new_col = []
    for i in range(len(col)):
        new_col.append(col[i])
    new_col[idx] = 'min_opp_dist1'
    new_col[idx+1] = 'mean_opp_dist1'
    new_col[idx+2] = 'std_opp_dist1'
    new_col[idx+3] = 'min_team_dist1'
    new_col[idx+4] = 'mean_team_dist1'
    new_col[idx+5] = 'std_team_dist1'

    new_col[idx+6] = 'min_opp_dist2'
    new_col[idx+7] = 'mean_opp_dist2'
    new_col[idx+8] = 'std_opp_dist2'
    new_col[idx+9] = 'min_team_dist2'
    new_col[idx+10] = 'mean_team_dist2'
    new_col[idx+11] = 'std_team_dist2'

    new_col[idx+12] = 'min_opp_dist3'
    new_col[idx+13] = 'mean_opp_dist3'
    new_col[idx+14] = 'std_opp_dist3'
    new_col[idx+15] = 'min_team_dist3'
    new_col[idx+16] = 'mean_team_dist3'
    new_col[idx+17] = 'std_team_dist3'

    return pd.DataFrame(np_pairs, columns=new_col)

def players_distances(original_dataframe):
    """
    Compute distances between each players for each pass
    :return: a two dim np matrix
    """
    # Get numpy version of the original dataset in order to extract position of each players
    x_np = original_dataframe.to_numpy()
    # Get the number of entries:
    n = x_np.shape[0]
    # Get position sub_array:
    p_x_pos = np.zeros((n, 22))
    p_y_pos = np.zeros((n, 22))
    for i in range(0, 22):
        p_x_pos[:, i] = x_np[:, 2 + i * 2]
        p_y_pos[:, i] = x_np[:, 3 + i * 2]

    # Store distances in a matrix:
    dist = np.zeros((n, 22, 22))
    # Get each players combinations:
    s = np.arange(22)
    j = np.arange(22)
    s, j = np.meshgrid(s, j)
    s = np.reshape(s, (-1, 1))
    j = np.reshape(j, (-1, 1))
    # For each pass:
    for i in range(0, x_np.shape[0]):
        tmp = np.sqrt(np.power(p_x_pos[i, s[:]] - p_x_pos[i, j[:]], 2) +
                          np.power(p_y_pos[i, s[:]] - p_y_pos[i, j[:]], 2))
        dist[i] = np.reshape(tmp, (22, 22))

    return dist

def compute_single_dist(arr):
    sender = arr[-1]
    arr = arr[:-1]
    p_x_pos, p_y_pos = np.hsplit(arr, 2)
    dist = np.zeros(22)
    for i in range(0, 22):
        dist[i] = np.sqrt((p_x_pos[i] - p_x_pos[int(sender-1)])**2 +
                          (p_y_pos[i] - p_y_pos[int(sender-1)])**2)
    return dist

def sender_players_distances(original_dataframe, pairs):
    """
    Compute distances between each players for each pass
    :return: a two dim np matrix
    """
    # Get numpy version of the original dataset in order to extract position of each players
    x_np = original_dataframe.to_numpy()
    # Get the number of entries:
    n = x_np.shape[0]
    # Get position sub_array:
    p_x_pos = np.zeros((n, 22))
    p_y_pos = np.zeros((n, 22))
    for i in range(0, 22):
        p_x_pos[:, i] = x_np[:, 2 + i * 2]
        p_y_pos[:, i] = x_np[:, 3 + i * 2]
    np_sender = original_dataframe['sender'].to_numpy()
    np_sender = np.reshape(np_sender, (n, 1))
    print(p_x_pos.shape)
    print(p_y_pos.shape)
    print(np_sender.shape)  
    tmp = np.hstack((p_x_pos, p_y_pos, np_sender))
    dist = np.apply_along_axis(compute_single_dist, 1, tmp)
    dist = np.repeat(dist, 22, axis=0)
    print(dist.shape)
    dist /= get_diagonale()
    for i in range(0,22):
        print(dist[:, i])
        pairs["dist_sender_{}".format(i+1)] = dist[:, i]
    return pairs

def normalizer(x):

    # Get numpy version
    x_np = x.to_numpy()
    x_np[:, 0] /= 22       # Sender
    x_np[:, 1] /= MAX_X_ABS_VAL    # Sender abs
    x_np[:, 2] /= MAX_Y_ABS_VAL    # Sender ordo
    x_np[:, 3] /= 22 # receiver
    x_np[:, 4] /= MAX_X_ABS_VAL #receiver abs
    x_np[:, 5] /= MAX_Y_ABS_VAL # receiver ordo
    x_np[:, 6] = (x_np[:, 6] - 0.5)*2 # same_team
    for i in range(9, 28):
        x_np[:, i] /= get_diagonale()
    return pd.DataFrame(x_np, columns=x.columns)


