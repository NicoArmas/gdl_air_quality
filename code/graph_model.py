import torch
from torch import nn
from tsl.nn.layers.graph_convs import GatedGraphNetwork
from tsl.nn.layers.temporal_attention import TemporalSelfAttention
from tsl.nn.utils.utils import get_functional_activation
from tsl.nn.blocks.encoders.mlp import MLP


class SpatialModel(GatedGraphNetwork):
    def __init__(self, hidden_size):
        super(SpatialModel, self).__init__(hidden_size, hidden_size)

    def forward(self, x, edge_index, mask=None):
        out = self.propagate(edge_index, x=x, mask=mask)
        out = self.update_mlp(torch.cat([out, x], -1)) + self.skip_conn(x)

        return out

    def message(self, x_i, x_j, mask_j):
        mij = self.msg_mlp(torch.cat([x_i, x_j], -1))
        return self.gate_mlp(mij) * mij


class AirQualityModel(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 window_size,
                 horizon,
                 num_heads=8,
                 n_sp_layers=1,
                 n_tp_layers=1):

        super(AirQualityModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = input_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.horizon = horizon

        self.n_sp_layers = n_sp_layers
        self.n_tp_layers = n_tp_layers

        self.activation = get_functional_activation('relu')
        self.upscale = nn.Linear(self.input_size, self.hidden_size)

        # spatial layers 1
        self.spatialSeq1 = nn.ModuleList()
        for _ in range(self.n_sp_layers):
            self.spatialSeq1.append(SpatialModel(self.hidden_size))

        # temporal layers
        self.temporalSeq = nn.ModuleList()
        for _ in range(self.n_tp_layers):
            self.temporalSeq.append(TemporalSelfAttention(self.hidden_size, self.num_heads))

        # spatial layers 2
        self.spatialSeq2 = nn.ModuleList()
        for _ in range(self.n_sp_layers):
            self.spatialSeq2.append(SpatialModel(self.hidden_size))

        self.augmented_size = self.hidden_size + self.input_size

        self.conv_layer = nn.Conv2d(self.window_size, self.horizon, 1)
        self.output_layer = MLP(self.augmented_size, self.hidden_size, self.output_size)

    def forward(self, x, edge_index=None, **kwargs):
        """
        The function takes in a 4D tensor (x) and returns a 4D tensor (output)
        
        :param x: [batch size, timestep, nodes, channels]
        :param edge_index: [2, num_edges]
        :return: The output of the model.
        """
        # x: [batch size, timestep, nodes, channels]
        input = self.upscale(x)
        input = self.activation(input)

        for i in range(self.n_sp_layers):
            input = self.spatialSeq1[i](input, edge_index)

        input = self.activation(input)

        for i in range(self.n_tp_layers):
            input, _ = self.temporalSeq[i](input)

        for i in range(self.n_sp_layers):
            input = self.spatialSeq2[i](input, edge_index)

        input = self.activation(input)

        input = torch.cat([input, x], dim=-1)
        output = self.conv_layer(input)
        output = self.output_layer(output)

        return output


if __name__ == '__main__':
    import tsl
    import torch
    import numpy as np

    from airquality import AirQuality as AQ

    dataset = AQ(is_subgraph=True, sub_start='6.0-79.0-8002.0', sub_size=70, data_dir='data')

    adj = dataset.get_connectivity(threshold=0.1,
                                   include_self=False,
                                   normalize_axis=1,
                                   layout="edge_index")

    from tsl.data import SpatioTemporalDataset

    torch_dataset = SpatioTemporalDataset(*dataset.numpy(return_idx=True),
                                          connectivity=adj,
                                          mask=dataset.mask,
                                          horizon=1,
                                          window=24)

    from tsl.data import SpatioTemporalDataModule
    from tsl.data.preprocessing import StandardScaler

    scalers = {'data': StandardScaler(axis=(0, 1))}

    splitter = dataset.get_splitter(val_len=0.1, test_len=0.2)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers=scalers,
        splitter=splitter,
        batch_size=32,
    )

    dm.setup()

    from tsl.nn.metrics.metrics import MaskedMAE, MaskedMAPE
    from tsl.predictors import Predictor

    loss_fn = MaskedMAE(compute_on_step=True)

    metrics = {'mae': MaskedMAE(compute_on_step=False)}

    model_kwargs = {
        'input_size': 1,
        'hidden_size': 32,
        'window_size': 24,  # era 12
        'horizon': 1
    }

    # setup predictor
    predictor = Predictor(
        model_class=AirQualityModel,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': 0.001},
        loss_fn=loss_fn,
        metrics=metrics
    )

    from pytorch_lightning.loggers import CSVLogger

    logger = CSVLogger(save_dir='models_data', name='graph_model_[0.001,32,24,1]')

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath='logs',
        save_top_k=1,
        monitor='val_mae',
        mode='min',
    )

    trainer = pl.Trainer(max_epochs=100,
                         logger=logger,
                         gpus=1 if torch.cuda.is_available() else None,
                         callbacks=[checkpoint_callback])

    trainer.fit(predictor, datamodule=dm)

    predictor.load_model(checkpoint_callback.best_model_path)
    predictor.freeze()

    performance = trainer.test(predictor, datamodule=dm)
