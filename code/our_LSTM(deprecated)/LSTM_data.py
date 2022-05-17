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
    It takes a dataset, a batch size, and a bptt length, and returns a list of batches, where each batch is a list of
    sequences, where each sequence is a list of symbols

    :param dataset: the dataset we want to batchify
    :param bptt_len: The length of the sequence of symbols that we will feed into the model
    :param batch_dim: the number of sequences in a batch
    :return: A list of lists of batches.
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
    It takes a dataframe, a batch dimension and a bptt_len (backpropagation through time length) and returns three batches
    of data, one for training, one for validation and one for testing

    :param df: the dataframe containing the data
    :param bptt_len: the length of the sequence we want to predict, defaults to 12 (optional)
    :param batch_dim: the number of nodes in each batch, defaults to 64 (optional)
    :param DEVICE: the device on which to run the model (CPU or GPU), defaults to cuda (optional)
    :return: batches_train, batches_valid, batches_test are lists of tuples. Each tuple contains a tensor of shape
    (batch_dim, bptt_len, nodes) and a tensor of shape (batch_dim, nodes)
    """


    ids = list(df.loc[:, 'ID'].drop_duplicates()) #get the unique IDs (lista di tutti gli id)
    # all_nodes = torch.empty(size=(0,0)) #create a tensor of the correct size
    # all_nodes = all_nodes.to(DEVICE)

    pm_25_tens = torch.Tensor(df['PM25'].values)
    df['PM25'] = normalize(pm_25_tens, dim=0) #normalizzo i valori di pm25 !(vanno normalizzati anche gli id!!)!


    for i, id in enumerate(ids): #per ogni id (enumerato)
        node_values = df.loc[df['ID'] == id, ['PM25']]  #prendo tutti i valori per quell id
        node_x_tens = np.array(node_values['PM25'].values) #tengo i valori e li metto in np array
        if i == 0:
            all_nodes = np.array([node_x_tens]) #inizializzo l np array in cui li vado a mettere
        else:
            all_nodes = np.vstack((all_nodes, [node_x_tens])) #altrimenti li stacko a quelli gia fatti

    #concludo avendo tutti i valori di pm25 stackati in np array

    all_nodes = torch.from_numpy(all_nodes).to(DEVICE) #trasferisco su tensore su GPU
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