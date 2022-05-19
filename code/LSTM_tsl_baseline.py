import torch
from torch import nn
import tsl
import numpy as np

from tsl.nn.blocks.encoders import RNN
from airquality import AirQuality as AQ

from tsl.data import SpatioTemporalDataset
from tsl.data import SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler

from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class TimeThenSpaceModel(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 rnn_layers,
                 output_size,
                 cell='lstm'):
        super(TimeThenSpaceModel, self).__init__()

        self.encoder = RNN(input_size=input_size,
                           hidden_size=hidden_size,
                           output_size=output_size,
                           n_layers=rnn_layers,
                           cell=cell)

    def forward(self, x, edge_index, edge_weight):
        input = self.encoder(x, return_last_state=True)
        input = input.view(input.shape[0], input.shape[1], input.shape[2], -1)
        input = input.view(input.shape[0], input.shape[2], input.shape[1], -1)
        return input

# full process for launching model with pytorch lightning

dataset = AQ(is_subgraph=True, sub_start='6.0-79.0-8002.0', sub_size=70, data_dir='../data')

adj = dataset.get_connectivity(threshold=0.1,
                               include_self=False,
                               normalize_axis=1,
                               layout="edge_index")


torch_dataset = SpatioTemporalDataset(*dataset.numpy(return_idx=True),
                                      connectivity=adj,
                                      mask=dataset.mask,
                                      horizon=12,
                                      window=12)

scalers = {'data': StandardScaler(axis=(0, 1))}

splitter = dataset.get_splitter(val_len=0.1, test_len=0.2)

dm = SpatioTemporalDataModule(
    dataset=torch_dataset,
    scalers=scalers,
    splitter=splitter,
    batch_size=64,
)

dm.setup()

from tsl.nn.metrics.metrics import MaskedMAE, MaskedMAPE
from tsl.predictors import Predictor

loss_fn = MaskedMAE(compute_on_step=True)

metrics = {'mae': MaskedMAE(compute_on_step=False),
           'mape': MaskedMAPE(compute_on_step=False)}

model_kwargs = {
    'input_size': 1,
    'hidden_size': 32,
    'rnn_layers': 1,
    'output_size': 12
}

# setup predictor
predictor = Predictor(
    model_class=TimeThenSpaceModel,
    model_kwargs=model_kwargs,
    optim_class=torch.optim.Adam,
    optim_kwargs={'lr': 0.001},
    loss_fn=loss_fn,
    metrics=metrics
)

logger = CSVLogger(save_dir='models_data', name='NOT CONSIDER')  # LSTM_model_hor12


checkpoint_callback = ModelCheckpoint(
    dirpath='logs',
    save_top_k=1,
    monitor='val_mae',
    mode='min',
)

trainer = pl.Trainer(max_epochs=50,
                     logger=logger,
                     gpus=1 if torch.cuda.is_available() else None,
                     callbacks=[checkpoint_callback])

trainer.fit(predictor, datamodule=dm)

predictor.load_model(checkpoint_callback.best_model_path)
predictor.freeze()

performance = trainer.test(predictor, datamodule=dm)
