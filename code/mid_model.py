import torch
from torch import nn
import tsl
import numpy as np

from tsl.nn.blocks.encoders import RNN
from tsl.nn.blocks.decoders import GCNDecoder


class TimeThenSpaceModel(torch.nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 rnn_layers,
                 gcn_layers,
                 horizon,
                 cell='lstm'):
        super(TimeThenSpaceModel, self).__init__()

        self.input_encoder = torch.nn.Linear(input_size, hidden_size)

        self.encoder = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers,
                           cell=cell)

        self.decoder = GCNDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=input_size,
            horizon=horizon,
            n_layers=gcn_layers
        )

    def forward(self, x, edge_index, edge_weight):
        # x: [batches steps nodes channels]
        x = self.input_encoder(x)

        x = self.encoder(x, return_last_state=True)

        return self.decoder(x, edge_index, edge_weight)


#from tsl.datasets import MetrLA

#dataset = MetrLA()
from airquality import AirQuality as AQ
dataset =AQ(is_subgraph=True, sub_start='6.0-73.0-1201.0', sub_size=100, data_dir='../data')

#print(dataset.dataframe())

adj = dataset.get_connectivity(threshold=0.1,
                               include_self=False,
                               normalize_axis=1,
                               layout="edge_index") # e qui che usa la matrice delle distanze

from tsl.data import SpatioTemporalDataset

torch_dataset = SpatioTemporalDataset(*dataset.numpy(return_idx=True),
                                      connectivity=adj,
                                      mask=dataset.mask,
                                      horizon=1,
                                      window=12)



from tsl.data import SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler

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
           'mape': MaskedMAPE(compute_on_step=False),
           'mae_at_15': MaskedMAE(compute_on_step=False, at=2),  # `2` indicated the third time step,
                                                                 # which correspond to 15 minutes ahead
           'mae_at_30': MaskedMAE(compute_on_step=False, at=5),
           'mae_at_60': MaskedMAE(compute_on_step=False, at=11), }

model_kwargs = {
    'input_size': 1,  # 1 channel #l input size e il numero di canali!
    'hidden_size': 32, #come trasformo (nota che lo fa per un canale mi conviene usar enumero piu grosso a me che ne ho 8!)
    'rnn_layers': 1,
    'gcn_layers': 2,
    'horizon': 1,  # 12, the number of steps ahead to forecast # lo faccio combaciare con quello settato prima
}


# setup predictor
predictor = Predictor(
    model_class=TimeThenSpaceModel,
    model_kwargs=model_kwargs,  #args passed to the model
    optim_class=torch.optim.Adam,
    optim_kwargs={'lr': 0.001},
    loss_fn=loss_fn,
    metrics=metrics
)


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath='logs',
    save_top_k=1,
    monitor='val_mae',
    mode='min',
)

trainer = pl.Trainer(max_epochs=100,
                     #logger=logger,
                     gpus=1 if torch.cuda.is_available() else None,
                     limit_train_batches=100,
                     callbacks=[checkpoint_callback])

trainer.fit(predictor, datamodule=dm)


predictor.load_model(checkpoint_callback.best_model_path)
predictor.freeze()

performance = trainer.test(predictor, datamodule=dm)




