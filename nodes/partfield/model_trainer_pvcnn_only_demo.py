import torch
from .model.triplane import TriplaneTransformer, get_grid_coord
from .model.model_utils import VanillaMLP
import torch.nn as nn
from .model.PVCNN.encoder_pc import TriPlanePC2Encoder, sample_triplane_feat


class Model(nn.Module):
    def __init__(self, cfg, init_device=None, dtype=None, operations=None):
        super().__init__()

        self.cfg = cfg
        self.triplane_resolution = cfg.triplane_resolution
        self.triplane_channels_low = cfg.triplane_channels_low
        self.triplane_transformer = TriplaneTransformer(
            input_dim=cfg.triplane_channels_low * 2,
            transformer_dim=1024,
            transformer_layers=6,
            transformer_heads=8,
            triplane_low_res=32,
            triplane_high_res=128,
            triplane_dim=cfg.triplane_channels_high,
            device=init_device, dtype=dtype, operations=operations,
        )
        self.sdf_decoder = VanillaMLP(input_dim=64,
                                      output_dim=1,
                                      out_activation="tanh",
                                      n_neurons=64,
                                      n_hidden_layers=6,
                                      device=init_device, dtype=dtype, operations=operations)
        self.use_pvcnn = cfg.use_pvcnnonly
        self.use_2d_feat = cfg.use_2d_feat
        if self.use_pvcnn:
            self.pvcnn = TriPlanePC2Encoder(
                cfg.pvcnn,
                device=init_device,
                dtype=dtype,
                operations=operations,
                shape_min=-1,
                shape_length=2,
                use_2d_feat=self.use_2d_feat)
        self.logit_scale = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.grid_coord = get_grid_coord(256)
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss(reduction='none')

        if cfg.regress_2d_feat:
            self.feat_decoder = VanillaMLP(input_dim=64,
                                output_dim=192,
                                out_activation="GELU",
                                n_neurons=64,
                                n_hidden_layers=6,
                                device=init_device, dtype=dtype, operations=operations)

 