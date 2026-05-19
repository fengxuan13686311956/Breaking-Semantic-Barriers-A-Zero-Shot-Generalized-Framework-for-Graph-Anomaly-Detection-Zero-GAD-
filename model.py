import torch.nn as nn
import torch.nn.functional as F


from utils import *
from torch import Tensor
from torch.nn.modules.module import Module
from torch_geometric.nn.inits import glorot
from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
from loss_func import sce_loss


def build_model(args):
    num_heads = args.num_heads_mae  # 4
    num_out_heads = args.num_out_heads_mae  # 1
    num_hidden = args.num_hidden_mae  # 256
    num_layers = args.num_layers_mae  # 2
    residual = args.residual_mae  # False
    attn_drop = args.attn_drop_mae  # 0.1
    in_drop = args.in_drop_mae  # 0.2
    norm = args.norm_mae  # None
    negative_slope = args.negative_slope_mae  # 0.2
    encoder_type = args.encoder_mae  # "gat"
    decoder_type = args.decoder_mae  # "gat"
    activation = args.activation_mae  # "prelu"
    loss_fn = args.loss_fn_mae  # "sce"
    alpha_l = args.alpha_l_mae  # 2
    concat_hidden = args.concat_hidden_mae  # False
    in_dim = args.in_dim_mae  # 8


    model = PreModel(
        in_dim=in_dim,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        norm=norm,
        loss_fn=loss_fn,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
    )
    return model


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            alpha_l: float = 2,
            concat_hidden: bool = False,
    ):
        super(PreModel, self).__init__()

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            activation=activation,
            dropout=feat_drop,
            nhead_out=dec_num_hidden,
            residual=residual,
            norm=norm,
            attn_drop=attn_drop,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=2,
            nhead=enc_nhead,
            activation=activation,
            dropout=feat_drop,
            nhead_out=dec_num_hidden,
            residual=residual,
            norm=norm,
            attn_drop=attn_drop,
        )

        self.hidden_output_layer = nn.Linear(dec_in_dim, 1)  # 输出概率
        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    def variance_loss(self,middle_feat):
        # middle_feat shape: [batch_size, num_nodes, feat_dim]
        mean_feat = torch.mean(middle_feat, dim=1, keepdim=True)  # 计算均值
        loss = torch.mean((middle_feat - mean_feat) ** 2)  # 计算均方差
        return loss

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, feat,adj):
        # ---- attribute reconstruction ----

        x=feat.clone()
        x.to("cuda:0")
        middle_feat = self.encoder(x,adj)

        # print(middle_feat[:1][0])
        # ---- attribute reconstruction ----
        middle_feat = self.encoder_to_decoder(middle_feat)
        loss_mid=self.variance_loss(middle_feat)
        # if self._decoder_type not in ("mlp", "linear"):
        #     # * remask, re-mask
        #     middle_feat[mask_nodes] = 0
        # if self._decoder_type in ("mlp", "liear"):
        #     recon = self.decoder(middle_feat)
        # else:

        recon= self.decoder(middle_feat,adj)

        return recon,loss_mid
