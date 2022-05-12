import torch
from torch import nn
from tsl.nn.layers.graph_convs import GatedGraphNetwork
from tsl.nn.layers.temporal_attention import TemporalSelfAttention
from tsl.nn.utils.utils import get_functional_activation
from tsl.nn.blocks.encoders.mlp import MLP

from tsl.nn.models import TransformerModel

import tsl
import torch
import numpy as np

from airquality import AirQuality as AQ

dataset = AQ(is_subgraph=True, sub_start='6.0-73.0-1201.0', sub_size=100, data_dir='../data')


adj = dataset.get_connectivity(threshold=0.1,
                               include_self=False,
                               normalize_axis=1,
                               layout="edge_index")

#serve matrice di adiacenza!
print(dataset.mask)

from tsl.data import SpatioTemporalDataset


torch_dataset = SpatioTemporalDataset(*dataset.numpy(return_idx=True),
                                      connectivity=adj,
                                      mask=dataset.mask,
                                      horizon=1,   #li metto gia da qui?
                                      window=24)   # li metto gia da qui? era 12

#torch_dataset.add_exogenous('mask', dataset.mask)


from tsl.data import SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler

scalers = {'data': StandardScaler(axis=(0, 1))}

splitter = dataset.get_splitter(val_len=0.1, test_len=0.2)

dm = SpatioTemporalDataModule(
    dataset=torch_dataset,
    scalers=scalers,
    splitter=splitter,
    batch_size=64,  #era 64!
)

dm.setup()

from tsl.nn.metrics.metrics import MaskedMAE, MaskedMAPE
from tsl.predictors import Predictor

loss_fn = MaskedMAE(compute_on_step=True)

metrics = {'mae': MaskedMAE(compute_on_step=False),
           'mape': MaskedMAPE(compute_on_step=False)}

model_kwargs = {
    'input_size': 1,
    'hidden_size': 63,   #era trentadue ma non andava per la divisibilita probabilemnte ce un problema con input_size_sp era 31
    'window_size': 24, #era 12
    'horizon': 1
}

# setup predictor
predictor = Predictor(
    model_class=TransformerModel,
    model_kwargs=model_kwargs,
    optim_class=torch.optim.Adam,
    optim_kwargs={'lr': 0.001}, #0.001
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
                    #limit_train_batches=100,
                     callbacks=[checkpoint_callback])

trainer.fit(predictor, datamodule=dm)

