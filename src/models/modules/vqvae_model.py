import torch
import numpy as np
import math
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import functools
import matplotlib.pyplot as plt
'''from architecture import Encoder
from architecture import Decoder
from architecture import VectorQuantizer'''
from src.models.modules.architecture import Encoder
from src.models.modules.architecture import Decoder
from src.models.modules.architecture import VectorQuantizer
from src.models.modules.ViewModule import View
import random

class vqvae_model_2d(nn.Module):
    def __init__(self,cfg):
        super(vqvae_model_2d, self).__init__()

        self.in_channels = cfg.in_channels
        self.num_hiddens = cfg.num_hiddens
        self.num_residual_hiddens = cfg.num_residual_hiddens
        self.num_residual_hiddens = cfg.num_residual_hiddens
        self.num_embeddings = cfg.num_embeddings
        self.embedding_dim = cfg.embedding_dim
        self.commitment_cost = cfg.commitment_cost

        self.encoder = Encoder(cfg)

        self.pre_vq_conv = nn.Conv2d(in_channels=self.num_hiddens,
                                      out_channels=self.embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self.vq_vae = VectorQuantizer(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        vq_loss, z_q, perplexity, _ = self.vq_vae(z)
        x_recon = self.decoder(z_q)


        outputs = {}
        outputs['vq_loss'] = vq_loss
        outputs['x_recon']= x_recon
        outputs['z_q'] = z_q
        outputs['perplexity'] = perplexity


        return outputs


