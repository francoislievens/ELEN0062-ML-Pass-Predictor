import numpy as np
import pandas as pd
#import cupy as cp

MAX_X_ABS_VAL = 5250
MAX_Y_ABS_VAL = 3400

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

def pass_distance(pairs):

    # check the first empty feature:
    col = pairs.columns
    idx = 0
    for i in range(0, len(col)):
        if 'feature_' in col[i]:
            idx = i
            break
    # Get numpy version
    np_pairs = pairs.to_numpy()

    # Compute distance
    np_pairs[:, idx] = np.sqrt(np.power(np_pairs[:, 1] - np_pairs[:, 4], 2) +
                               np.power(np_pairs[:, 2] - np_pairs[:, 5], 2))
    # Update headers
    new_col = []
    for i in range(len(col)):
        if i == idx:
            new_col.append('distance')
        else:
            new_col.append(col[i])
    return pd.DataFrame(np_pairs, columns=new_col)

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

        tmp_opp_dist_sum = []
        tmp_team_dist_sum = []

        for j in range(0, 22):
            # Don't look at sender or receiver
            if j != sender and j != rec:
                # Same team
                if j < 11 and rec < 11 or j >= 11 and rec >= 11:
                    tmp_team_dist_sum.append(sender_dists[j] + rec_dists[j])
                # Opposit team
                if j < 11 and rec >= 11 or j >= 11 and rec < 11:
                    tmp_opp_dist_sum.append(sender_dists[j] + rec_dists[j])


        np_pairs[i, idx] = np.min(tmp_opp_dist_sum)
        np_pairs[i, idx+1] = np.mean(tmp_opp_dist_sum)
        np_pairs[i, idx+2] = np.std(tmp_opp_dist_sum)
        np_pairs[i, idx+3] = np.min(tmp_team_dist_sum)
        np_pairs[i, idx+4] = np.mean(tmp_team_dist_sum)
        np_pairs[i, idx+5] = np.std(tmp_team_dist_sum)

    # Update headers
    new_col = []
    for i in range(len(col)):
        new_col.append(col[i])
    new_col[idx] = 'min_opp_dist'
    new_col[idx+1] = 'mean_opp_dist'
    new_col[idx+2] = 'std_opp_dist'
    new_col[idx+3] = 'min_team_dist'
    new_col[idx+4] = 'mean_team_dist'
    new_col[idx+5] = 'std_team_dist'

    return pd.DataFrame(np_pairs, columns=new_col)

def dist_tool_old(pairs, dist):
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

        tmp_opp_dist_sum = []
        tmp_team_dist_sum = []

        for j in range(0, 22):
            for k in range(0, 22):
                if j != sender and j != rec and k != sender and k != rec:
                    # Same team:
                    if j < 11 and team == 0 and k < 11 or j >= 11 and team == 1 and k >= 11:
                        tmp_team_dist_sum.append(sender_dists[j] + rec_dists[k])
                    # Opposite team:
                    if j < 11 and team == 1 and k < 11 or j >= 11 and team == 0 and k >= 11:
                        tmp_opp_dist_sum.append(sender_dists[j] + rec_dists[k])


        np_pairs[i, idx] = np.min(tmp_opp_dist_sum)
        np_pairs[i, idx+1] = np.mean(tmp_opp_dist_sum)
        np_pairs[i, idx+2] = np.std(tmp_opp_dist_sum)
        np_pairs[i, idx+3] = np.min(tmp_team_dist_sum)
        np_pairs[i, idx+4] = np.mean(tmp_team_dist_sum)
        np_pairs[i, idx+5] = np.std(tmp_team_dist_sum)

    # Update headers
    new_col = []
    for i in range(len(col)):
        new_col.append(col[i])
    new_col[idx] = 'min_opp_dist'
    new_col[idx+1] = 'mean_opp_dist'
    new_col[idx+2] = 'std_opp_dist'
    new_col[idx+3] = 'min_team_dist'
    new_col[idx+4] = 'mean_team_dist'
    new_col[idx+5] = 'std_team_dist'

    return pd.DataFrame(np_pairs, columns=new_col)

def dist_tool_vc(pairs, dist):
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

    print('START')

    # Vectorize sub function:
    v_dist_tool = np.vectorize(pairs_dist_tool, otypes=[float, float, float, float, float, float], excluded=['dist'])
    # Get arrays for vecto
    senders_v = np_pairs[:, 0]
    rec_v = np_pairs[:, 3]
    pass_idx_v = np_pairs[:, 7]
    # Compute data
    np_pairs[:, idx:idx+6] = np.reshape(v_dist_tool(sdr=senders_v, rcv=rec_v, p_id=pass_idx_v, dist=dist),
                                        (np_pairs.shape[0], 6))

    print('END')
    # Update headers
    new_col = []
    for i in range(len(col)):
        new_col.append(col[i])
    new_col[idx] = 'min_opp_dist'
    new_col[idx+1] = 'mean_opp_dist'
    new_col[idx+2] = 'std_opp_dist'
    new_col[idx+3] = 'min_team_dist'
    new_col[idx+4] = 'mean_team_dist'
    new_col[idx+5] = 'std_team_dist'

    return pd.DataFrame(np_pairs, columns=new_col)

def pairs_dist_tool(sdr, rcv, p_id, dist):

    sender = int(sdr - 1)
    rec = int(rcv - 1)
    pass_idx = int(p_id)
    # Get distances
    sender_dists = dist[pass_idx, sender, :]
    rec_dists = dist[pass_idx, rec, :]
    team = 0
    if rec >= 11:
        team = 1

    tmp_opp_dist_sum = []
    tmp_team_dist_sum = []

    for j in range(0, 22):
        # Don't look at sender or receiver
        if j != sender and j != rec:
            # Same team
            if j < 11 and rec < 11 or j >= 11 and rec >= 11:
                tmp_team_dist_sum.append(sender_dists[j] + rec_dists[j])
            # Opposit team
            if j < 11 and rec >= 11 or j >= 11 and rec < 11:
                tmp_opp_dist_sum.append(sender_dists[j] + rec_dists[j])

    a = np.min(tmp_opp_dist_sum)
    b = np.mean(tmp_opp_dist_sum)
    c = np.std(tmp_opp_dist_sum)
    d = np.min(tmp_team_dist_sum)
    e = np.mean(tmp_team_dist_sum)
    f = np.std(tmp_team_dist_sum)

    return a, b, c, d, e, f



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

def normalizer(x):

    # Get numpy version
    x_np = x.to_numpy()
    x_np[:, 0] /= 22       # Sender
    x_np[:, 1] /= MAX_X_ABS_VAL    # Sender abs
    x_np[:, 2] /= MAX_Y_ABS_VAL    # Sender ordo
    x_np[:, 3] /= 22
    x_np[:, 4] /= MAX_X_ABS_VAL
    x_np[:, 5] /= MAX_Y_ABS_VAL
    for i in range(9, 16):
        x_np[:, i] /= MAX_X_ABS_VAL
    return pd.DataFrame(x_np, columns=x.columns)

def gravity_center(pairs, orig_data):
    """
    Compute distance between sender and receiver + dist between each opposant and the sender +
    dist between the same opposant and the receiver. The shoortest dist is chosen
    Compute Mean and min
    """
    # Get the number of entries:
    x_np = orig_data.to_numpy()
    n = x_np.shape[0]
    # Get position sub_array:
    x_pos = np.zeros((n, 22))
    y_pos = np.zeros((n, 22))
    for i in range(0, 22):
        x_pos[:, i] = x_np[:, 2 + i * 2]
        y_pos[:, i] = x_np[:, 3 + i * 2]
    # check the first empty feature:
    col = pairs.columns
    idx = 0
    for i in range(0, len(col)):
        if 'feature_' in col[i]:
            idx = i
            break
    # Get numpy version
    np_pairs = pairs.to_numpy()

    # Compute:
    for i in range(0, np_pairs.shape[0]):
        if np_pairs[i, 0] < 11:
            np_pairs[i, idx] = np.mean(x_pos[0:11])
            np_pairs[i, idx+1] = np.mean(y_pos[0:11])
            np_pairs[i, idx+2] = np.mean(x_pos[11:22])
            np_pairs[i, idx+3] = np.mean(y_pos[11:22])
        else:
            np_pairs[i, idx] = np.mean(x_pos[11:22])
            np_pairs[i, idx+1] = np.mean(y_pos[11:22])
            np_pairs[i, idx+2] = np.mean(x_pos[0:11])
            np_pairs[i, idx+3] = np.mean(y_pos[0:11])

    # Update headers
    new_col = []
    for i in range(len(col)):
        new_col.append(col[i])
    new_col[idx] = 'sender_team_gravity_x'
    new_col[idx+1] = 'sender_team_gravity_y'
    new_col[idx+2] = 'sender_opp_gravity_x'
    new_col[idx+3] = 'sender_opp_gravity_y'

    return pd.DataFrame(np_pairs, columns=new_col)

