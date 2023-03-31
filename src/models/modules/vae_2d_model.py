import torch
import numpy as np
import math
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import functools
from src.models.modules.unifiedArchitecture_vae import Encoder_unified
from src.models.modules.unifiedArchitecture_vae import Decoder_unified
from src.models.modules.ViewModule import View


### Variational Autoencoder 2D ###
class variational_Autoencoder_2D(nn.Module):
    def __init__(self, cfg):
        super(variational_Autoencoder_2D, self).__init__()

        self.imageDim = cfg.imageDim  # Input Dimension of the Image
        self.kernelSize = cfg.kernelSize  # KernelSize
        self.latentSize = cfg.latentSize
        self.channelsEnc = cfg.fmapsEnc
        self.channelsDec = cfg.fmapsDec
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(cfg.dropRate)
        self.interRes = cfg.interRes  # [HxWxD]
        self.bottleneckFeatures = cfg.bottleneckFmaps

        if cfg.EncoderArch == 'unified':
            self.encoderUnified = Encoder_unified(cfg)


        # Bottleneck mu
        self.conv_enc_btn1_mu = nn.Conv2d(in_channels=self.channelsEnc[-1],
                                          out_channels=self.bottleneckFeatures,
                                          kernel_size=1,
                                          stride=1,
                                          bias=False)
        self.BN_enc_btn1_mu = nn.BatchNorm2d(num_features=self.bottleneckFeatures)
        self.conv_enc_btn2_mu = nn.Conv2d(in_channels=self.bottleneckFeatures,
                                          out_channels=self.latentSize,
                                          kernel_size=[self.interRes[0], self.interRes[1]],
                                          stride=1,
                                          bias=False)
        self.flatten_enc_btn_mu = nn.Flatten(start_dim=1, end_dim=-1)


        self.fc_enc_btn_mu = nn.Linear(self.interRes[0]*self.interRes[1]*self.bottleneckFeatures, self.latentSize,bias=True)


        # Bottleneck sigma
        self.conv_enc_btn1_logvar = nn.Conv2d(in_channels=self.channelsEnc[-1],
                                              out_channels=self.bottleneckFeatures,
                                              kernel_size=1,
                                              stride=1,
                                              bias=False)
        self.BN_enc_btn1_logvar = nn.BatchNorm2d(num_features=self.bottleneckFeatures)
        self.flatten_enc_btn_logvar = nn.Flatten(start_dim=1, end_dim=-1)

        self.fc_enc_btn_logvar = nn.Linear(
            self.interRes[0] * self.interRes[1] * self.bottleneckFeatures, self.latentSize,
            bias=True)  # produces logvar

        ### build the decoder ###

        self.fc_dec_btn = nn.Linear(self.latentSize,
                                    self.bottleneckFeatures * self.interRes[0] * self.interRes[1])
        self.view_dec = View([-1, self.bottleneckFeatures, self.interRes[0], self.interRes[1]])
        self.BN_dec_btn0 = nn.BatchNorm2d(self.bottleneckFeatures)
        self.conv_dec_btn1 = nn.Conv2d(in_channels=self.bottleneckFeatures,
                                       out_channels=self.channelsDec[0],
                                       kernel_size=1,
                                       stride=1,
                                       bias=False)
        self.BN_dec_btn1 = nn.BatchNorm2d(self.channelsDec[0])

        # unified decoder
        self.decoderUnified = Decoder_unified(cfg)

        # Upsampling to outputsize
        self.conv_dec_final = nn.Conv2d(in_channels=self.channelsDec[-1],
                                        out_channels=1,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # unified network architecture encoder [Baur et al.]

        x = self.encoderUnified(x)

        # Bottleneck (btn) close
        # mu
        x_mu = self.conv_enc_btn1_mu(x)
        x_mu = self.BN_enc_btn1_mu(x_mu)
        x_mu = self.activation(x_mu)
        # x_mu = self.conv_enc_btn2_mu(x_mu)
        x_mu = self.flatten_enc_btn_mu(x_mu)
        # logvar
        x_logvar = self.conv_enc_btn1_logvar(x)
        x_logvar = self.BN_enc_btn1_logvar(x_logvar)
        x_logvar = self.activation(x_logvar)
        # x_logvar = self.conv_enc_btn2_logvar(x_logvar) # in Baurs arch this is replaced by the flatten and the a FC NN is used to bring it to latentspace size.
        x_logvar = self.flatten_enc_btn_logvar(x_logvar)

        # latent Space distribution
        x_mu = self.fc_enc_btn_mu(x_mu)
        x_logvar = self.fc_enc_btn_logvar(x_logvar)
        z_sampled = self.reparameterize(x_mu, x_logvar)

        # Bottleneck (btn) open
        y = self.fc_dec_btn(z_sampled)
        y = self.dropout(y)
        y = self.view_dec(y)
        y = self.BN_dec_btn0(y)
        y = self.activation(y)
        y = self.conv_dec_btn1(y)
        y = self.BN_dec_btn1(y)
        y = self.activation(y)

        # Unified network architecture decoder
        y = self.decoderUnified(y)

        x_hat = self.conv_dec_final(y)
        outputs = {}
        outputs['x_hat'] = x_hat  # recontsructed image
        outputs['mu'] = x_mu
        outputs['logvar'] = x_logvar
        outputs['z'] = z_sampled
        return outputs
