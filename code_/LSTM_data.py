import pandas as pd
from torch import Tensor
from torch.nn.functional import normalize
from torch import split

# initialise data of lists.
data = {'id': [i for i in range(1000)],
        'PM25': [i for i in range(0, 10000, 10)],
        'PM25_label': [i for i in range(0, 20000, 20)]}

# Create DataFrame
df = pd.DataFrame(data)


def batchify(features, label, batch_dim):
    """
    It takes a list of features and a list of labels, and returns a list of tuples of features and labels, where each tuple
    is a batch

    :param features: the features of the data
    :param label: the label of the data
    :param batch_dim: The number of batches to split the data into
    :return: A list of tuples, where each tuple is a batch of features and labels.
    """
    ft = split(features, batch_dim)
    lbl = split(label, batch_dim)
    batches = []

    for i in range(len(ft)):
        batches.append((ft[i], lbl[i]))

    return batches


# convert pandas to tensor (index is cut out automatically)
def prepare_data(df, batch_dim=64, DEVICE='cuda'):
    tens = Tensor(df.values)
    tens = tens.to(DEVICE)
    X, Y = tens[:, :-1], tens[:, -1]
    X = normalize(X)  # normalize per columns
    # training 0.7, val 0.15, test 0.15
    rows = tens.shape[0]
    func = lambda perc: rows // 100 * perc
    train = func(70)
    valid = func(15)
    train_ft, valid_ft, test_ft = X[:train], X[train + 1: train + valid + 1], X[train + valid + 1:]
    train_lbl, valid_lbl, test_lbl = Y[:train], Y[train + 1: train + valid + 1], Y[train + valid + 1:]
    # now create batches
    batches_train = batchify(train_ft, train_lbl, batch_dim)
    batches_valid = batchify(valid_ft, valid_lbl, batch_dim)
    batches_test = batchify(test_ft, test_lbl, batch_dim)

    return batches_train, batches_valid, batches_test

batches_train, batches_valid, batches_test = prepare_data(df)