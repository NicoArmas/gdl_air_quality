import os
import torch
import torch.nn as nn
from torch.nn.modules.activation import Softmax
import torch.optim as optim
import LSTM_data as ld
import pandas as pd
from torch.nn import L1Loss
import pickle


# We have an embedding layer, an LSTM layer, and a fully connected layer
class LSTMModel(nn.Module):

    def __init__(self, hidden_size, lstm_dim, num_layers, window, input_dim, output_dim):
        """
        The function takes in the input dimension, the hidden dimension, the LSTM dimension, the number of layers, and the
        output dimension. It then creates an embedding layer, an LSTM layer, and a fully connected layer

        :param hidden_size: the number of features in the hidden state h
        :param lstm_dim: the number of hidden units in the LSTM cell
        :param num_layers: number of layers in the LSTM
        :param input_dim: the dimension of the input vector
        :param output_dim: the number of classes we have, two in our case (pos/neg)
        """
        super(LSTMModel, self).__init__()
        # RIDEFINIRLA IN BASE AL LAVORO FATTO!

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.lstm_dim = lstm_dim
        self.window = window
        self.num_layers = num_layers
        self.upscale = nn.Linear(input_dim, hidden_size)
        self.LSTM = nn.LSTM(hidden_size, lstm_dim, num_layers)
        self.flc = nn.Linear(lstm_dim, output_dim)

    def forward(self, x, h, c):
        """
        The function takes in the input, the hidden state and the cell state, and returns the output and the hidden state
        and the cell state

        :param x: input to the LSTM
        :param h: hidden state
        :param c: the cell state
        :return: The output of the LSTM layer, and the hidden and cell states of the LSTM layer.
        """
        upscaled = self.upscale(x)
        outputs, (h1, c1) = self.LSTM(upscaled, (h, c))
        outputs = self.flc(outputs)
        (h1, c1) = (h1.detach(), c1.detach())
        return outputs, (h1, c1)


# Trainer
# It takes a model, a loss function, and an optimizer, and trains the model on the data
class Trainer():

    def __init__(self, model, loss_fn, loss_check, optimizer):
        """
        The `__init__` function takes these parameters and sets them as attributes of the `Trainer` object.

        So, when we call `self.model`, we're actually calling the model that we passed in as a parameter.

        The same goes for `self.loss_fn` and `self.optimizer`

        :param model: The model we're training
        :param loss_fn: This is the loss function that we want to minimize
        :param optimizer: This is the algorithm we'll use to update the parameters of the module. Here we use SGD
        """
        self.model = model
        self.loss_fn = loss_fn
        self.loss_check = loss_check
        self.optimizer = optimizer

    def train(self, train_x):
        """
        We take the input data, pass it through the model, calculate the loss, backpropagate the loss, and update the model
        parameters

        :param train_x: a list of batches of input data
        :return: The loss of the model
        """  # self.model.hidden_size
        h = torch.zeros(self.model.num_layers, 32, self.model.lstm_dim, device=DEVICE)
        c = torch.zeros(self.model.num_layers, 32, self.model.lstm_dim, device=DEVICE)

        for batch in train_x:  # train_x contains all the batches
            batch = batch.to(DEVICE)
            batch = batch.view(batch.shape[0], batch.shape[-1], -1)
            self.model.train()
            self.optimizer.zero_grad()
            output, (h, c) = self.model(batch[:-1], h, c)
            # a = output.flatten(end_dim=-2)
            # b = batch[1:].flatten()
            # loss = self.loss_check(output.flatten(end_dim=-2), batch[1:].flatten())
            #print(f' output modello e{output[-1, :, 32]}')
            #print(f'target e {batch[-1, :, 32]}')
            loss = self.loss_check(output, batch[1:])

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.optimizer.step()

        return loss


def training():
    """
    > The function trains the model until the perplexity is less than 1.03
    """
    perp = 2 ** 10
    i = 0
    while perp >= 0.5:
        temp = engine.train(batches)  # dobbiamo crearli con dataloaders
        i += 1
        print(f'epoch {i}')
        print(f"Model perplexity is {temp}")

        if temp < perp:
            perp = temp

    print('finished training')


def predict():

    h = torch.zeros(1, 32, 2048, device=DEVICE)
    c = torch.zeros(1, 32, 2048, device=DEVICE)
    model.eval()
    for batch in batches_test:
        batch = batch.to(DEVICE)
        batch = batch.view(batch.shape[0], batch.shape[-1], -1)
        with torch.no_grad():
            output, (h, c) = model(batch[:-1], h, c)
            print(f' output modello e{output[-1, :, 32]}')
            print(f'target e {batch[-1, :, 32]}')
            loss = loss_check(output, batch[1:])
            print(f' la loss e {loss}')





# prova
# data = {'id': [i for i in range(1000)],
#         'PM25': [i for i in range(0, 10000, 10)],
#         'PM25_label': [i for i in range(0, 20000, 20)]}

# Create DataFrame
file = open("LSTM_batches.pickle", "rb")
batches = pickle.load(file)
file.close()

batches_train, batches_test = batches[:61], batches[61:]
# DEVI NORMALIZZARLI!

print('data splitted')

DEVICE = 'cuda'

# self, upscale_size, lstm_dim, num_layers, window, input_dim, output_dim

# Hyper-params:
batches = batches_train  # proviamo cosi per ora
upscale_size = 256
lstm_dim = 2048
num_layers = 1
window = 12
learning_rate = 0.001  # 0.001 0.002, 0.00075
input_dim = 70
output_dim = 70

model = LSTMModel(upscale_size, lstm_dim, num_layers, window, input_dim, output_dim)
model = model.to(DEVICE)

print('model defined')

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
loss_check = L1Loss()
engine = Trainer(model, loss_fn, loss_check, optimizer)
print('starting training...')
training()

print('try some testing')
predict()


print("all done!")
