
def dist_team(pairs, dist):

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
        sender_op_dist = None
        # check the sender team
        if sender < 11:
            sender_op_dist = dist[pass_idx, sender, :11]
            rec_op_dist = dist[pass_idx, rec, :11]
        else:
            sender_op_dist = dist[pass_idx, sender, 11:]
            rec_op_dist = dist[pass_idx, rec, 11:]
        sum_matrix = np.ones((11, 11))
        for j in range(0, 11):
            sum_matrix[j, :] *= sender_op_dist[j]
        for j in range(0, 11):
            sum_matrix[:, j] += rec_op_dist[j]

        np_pairs[i, idx] = np.min(sum_matrix)
        np_pairs[i, idx+1] = np.mean(sum_matrix)

    # Update headers
    new_col = []
    for i in range(len(col)):
        if i == idx:
            new_col.append('min_dist_opp')
        if i == idx + 1:
            new_col.append('mean_dist_opp')
        else:
            new_col.append(col[i])
    return pd.DataFrame(np_pairs, columns=new_col)

def avg_dist_teammates(pairs, dist):

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
        sender = int(np_pairs[i, 3])
        if sender <= 11:
            np_pairs[i, idx] = np.mean(dist[int(np_pairs[i, 7]), sender-1, :11])
        else:
            np_pairs[i, idx] = np.mean(dist[int(np_pairs[i, 7]), sender-1, 11:])

    # Update headers
    new_col = []
    for i in range(len(col)):
        if i == idx:
            new_col.append('avg_dist_teammates')
        else:
            new_col.append(col[i])
    return pd.DataFrame(np_pairs, columns=new_col)

def avg_dist_opp(pairs, dist):

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
        sender = int(np_pairs[i, 3])
        if sender <= 11:
            np_pairs[i, idx] = np.mean(dist[int(np_pairs[i, 7]), sender-1, 11:])
        else:
            np_pairs[i, idx] = np.mean(dist[int(np_pairs[i, 7]), sender-1, :11])

    # Update headers
    new_col = []
    for i in range(len(col)):
        if i == idx:
            new_col.append('avg_dist_opp')
        else:
            new_col.append(col[i])
    return pd.DataFrame(np_pairs, columns=new_col)
