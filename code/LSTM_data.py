import pandas as pd
import torch
from torch.nn.functional import normalize
import numpy as np

# initialise data of lists.
# data = {'id': [i for i in range(10)],
#         'PM25': [i for i in range(0, 10000, 10)],
#         'PM25_label': [i for i in range(0, 20000, 20)]}

# # Create DataFrame
# df = pd.DataFrame(data)


def batchify(dataset, bptt_len, batch_dim):
    """
    It takes a list of features and a list of labels, and returns a list of tuples of features and labels, where each tuple
    is a batch

    :param features: the features of the data
    :param label: the label of the data
    :param batch_dim: The number of batches to split the data into
    :return: A list of tuples, where each tuple is a batch of features and labels.
    """
    all_node_batches = []

    for i in range(dataset.shape[1]):
        batches = []
        data = dataset[:, i]
        mis_len = len(data)
        segment_len = mis_len // batch_dim + 1
        num_batches = segment_len // bptt_len + 1
        
        for i in range(num_batches):
            # Prepare batches such that the last symbol of the current batch
            # is the first symbol of the next batch.

                if i == 0:
                    batch = data[:(i+1) * bptt_len]
                else:
                    batch = data[i * bptt_len - 1:(i + 1) * bptt_len]
                
                batches.append(batch)
    
        all_node_batches.append(batches)

    return all_node_batches


# convert pandas to tensor (index is cut out automatically)
def prepare_data(df, bptt_len = 12, batch_dim=64, DEVICE='cuda'):
    """
    It takes a dataframe, converts it to a tensor, normalizes it, splits it into train, validation and test sets, and then
    creates batches for each of the sets

    :param df: the dataframe
    :param batch_dim: the number of rows in each batch, defaults to 64 (optional)
    :param DEVICE: the device to use for training, defaults to cuda (optional)
    :return: batches_train, batches_valid, batches_test
    """

    ids = list(df.loc[:, 'ID'].drop_duplicates()) #get the unique IDs
    # all_nodes = torch.empty(size=(0,0)) #create a tensor of the correct size
    # all_nodes = all_nodes.to(DEVICE)

    pm_25_tens = torch.Tensor(df['PM25'].values)
    df['PM25'] = normalize(pm_25_tens, dim=0)


    for i, id in enumerate(ids):
        node_values = df.loc[df['ID'] == id, ['PM25']]
        node_x_tens = np.array(node_values['PM25'].values)
        if i == 0:
            all_nodes = np.array([node_x_tens])
        else:
            all_nodes = np.vstack((all_nodes, [node_x_tens]))

    all_nodes = torch.from_numpy(all_nodes).to(DEVICE)
    all_nodes = torch.transpose(all_nodes, 1, 0) # new shape: (rows: misurations, cols: nodes)
    
    length = all_nodes.shape[0]

    # training 0.70, val 0.15, test 0.15
    
    func = lambda perc: length // 100 * perc
    train = func(70)
    valid = func(15)

    train_data = all_nodes[:train]
    valid_data = all_nodes[train:train + valid]
    test_data = all_nodes[train + valid:]

    # bptt_len can be seen as our time window


    batches_train = batchify(train_data, bptt_len, batch_dim) # N.B. our batches have info on nodes, so one more dimension to iterate on
    batches_valid = batchify(valid_data, bptt_len, batch_dim) 
    batches_test = batchify(test_data, bptt_len, batch_dim)

    return batches_train, batches_valid, batches_test

    #######################################################################################################################

    # tens = Tensor(df.values)
    # tens = tens.to(DEVICE)
    # X, Y = tens[:, :-1], tens[:, -1]
    # X = normalize(X)  # normalize per columns
    # # training 0.7, val 0.15, test 0.15
    # rows = tens.shape[0]
    # func = lambda perc: rows // 100 * perc
    # train = func(70)
    # valid = func(15)
    # train_ft, valid_ft, test_ft = X[:train], X[train + 1: train + valid + 1], X[train + valid + 1:]
    # train_lbl, valid_lbl, test_lbl = Y[:train], Y[train + 1: train + valid + 1], Y[train + valid + 1:]
    # # now create batches
    # batches_train = batchify(train_ft, train_lbl, batch_dim)
    # batches_valid = batchify(valid_ft, valid_lbl, batch_dim)
    # batches_test = batchify(test_ft, test_lbl, batch_dim)

    # return batches_train, batches_valid, batches_test

# batches_train, batches_valid, batches_test = prepare_data(df)