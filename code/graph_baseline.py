import torch
from torch import nn
from tsl.nn.layers.graph_convs import GatedGraphNetwork
from tsl.nn.layers.temporal_attention import TemporalSelfAttention
from tsl.nn.utils.utils import get_functional_activation
from tsl.nn.blocks.encoders.mlp import MLP

from tsl.nn.models import TCNModel

import tsl
import torch
import numpy as np

from airquality import AirQuality as AQ
from tsl.data import SpatioTemporalDataset

from tsl.data import SpatioTemporalDataModule
from tsl.data.preprocessing import StandardScaler

from tsl.nn.metrics.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from tsl.predictors import Predictor

from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Dilated convolutional network
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
                                      window=24)

scalers = {'data': StandardScaler(axis=(0, 1))}

splitter = dataset.get_splitter(val_len=0.1, test_len=0.2)

dm = SpatioTemporalDataModule(
    dataset=torch_dataset,
    scalers=scalers,
    splitter=splitter,
    batch_size=512,
)

dm.setup()

loss_fn = MaskedMAE(compute_on_step=True)

metrics = {'mae': MaskedMAE(compute_on_step=False)}

model_kwargs = {
    'input_size': 1,
    'hidden_size': 32,
    'ff_size': 32,
    'output_size': 1,
    'horizon': 12,
    'kernel_size': 2,
    'n_layers': 4,
    'exog_size': 0
}

# setup predictor
predictor = Predictor(
    model_class=TCNModel,
    model_kwargs=model_kwargs,
    optim_class=torch.optim.Adam,
    optim_kwargs={'lr': 0.001},  # 0.001
    loss_fn=loss_fn,
    metrics=metrics
)

logger = CSVLogger(save_dir='models_data', name='baseline_graph_hor12')

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
