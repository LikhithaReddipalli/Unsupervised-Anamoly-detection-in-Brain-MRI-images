import torch
import numpy as np
import math
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import functools

# Adapted implementation from Zalando research
# https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=m9kl-tLulY-F


# residual block
class Residual(nn.Module):
    def __init__(self,in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()

        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)

# residual stack which has residual blocks
class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self.num_residual_layers = num_residual_layers

        self.layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self.num_residual_layers)])

    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
        return F.relu(x)

# Encoder block
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.in_channels = config['in_channels'] # number of feature maps for encoder
        self.num_hiddens = config['num_hiddens']
        self.num_residual_hiddens= config['num_residual_hiddens'] # number of feature maps in residual blocks
        self.num_residual_layers = config['num_residual_layers'] # number of residual layers

        self.conv_1 = nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.num_hiddens // 2,
                                kernel_size=4,
                                stride=2, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=self.num_hiddens // 2,
                                out_channels=self.num_hiddens,
                                kernel_size=4,
                                stride=2, padding=1)
        self.conv_3 = nn.Conv2d(in_channels=self.num_hiddens,
                                out_channels=self.num_hiddens,
                                kernel_size=3,
                                stride=1, padding=1)
        self.residual_stack = ResidualStack(in_channels=self.num_hiddens,
                                             num_hiddens=self.num_hiddens,
                                             num_residual_layers=self.num_residual_layers,
                                             num_residual_hiddens=self.num_residual_hiddens)

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = F.relu(x)

        x = self.conv_2(x)
        x = F.relu(x)

        x = self.conv_3(x)
        return self.residual_stack(x)

# decoder block
class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.in_channels = config['embedding_dim'] # size of codebook vectors
        self.num_hiddens = config['num_hiddens']
        self.num_residual_hiddens= config['num_residual_hiddens']
        self.num_residual_layers = config['num_residual_layers']

        self.conv_1 = nn.Conv2d(in_channels=self.in_channels,
                                out_channels=self.num_hiddens,
                                kernel_size=3,
                                stride=1, padding=1)

        self.residual_stack = ResidualStack(in_channels=self.num_hiddens,
                                            num_hiddens=self.num_hiddens,
                                            num_residual_layers=self.num_residual_layers,
                                            num_residual_hiddens=self.num_residual_hiddens)

        self.conv_trans_1 = nn.ConvTranspose2d(in_channels=self.num_hiddens,
                                                out_channels=self.num_hiddens // 2,
                                                kernel_size=3,
                                                stride=2, padding=1)

        self.conv_trans_2 = nn.ConvTranspose2d(in_channels=self.num_hiddens // 2,
                                                out_channels=1,
                                                kernel_size=3,
                                                stride=2, padding = 1)

    def forward(self, inputs):
        x = self.conv_1(inputs)

        x = self.residual_stack(x)

        x = self.conv_trans_1(x)
        x = F.relu(x)

        return self.conv_trans_2(x)


# Vector quantizer, where codebook vectors are updated

class VectorQuantizer(nn.Module):
    def __init__(self, config):
        super(VectorQuantizer, self).__init__()

        self.num_embeddings = config['num_embeddings']
        self.embedding_dim = config['embedding_dim']
        self.commitment_cost =config['commitment_cost']

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # num_embeddings – size of the dictionary of embeddings
        # embedding_dim – the size of embedding vector
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)# initialize the vector weights





    def forward(self,z):
            """
                    Inputs the output of the encoder network z and maps it to a discrete
                    one-hot vector that is the index of the closest embedding vector e_j
                    z (continuous) -> z_q (discrete)

                    (z= input)
                    z.shape = (batch, channel, height, width)
                    z.shape = (batch, channel, height, width)
                    quantization pipeline:
                        1. get encoder input (B,C,H,W)
                        2. flatten input to (B*H*W,C)
                    """
            # reshape z -> (batch, height, width, channel) and flatten

            # convert inputs from BCHW -> BHWC
            z = z.permute(0,2,3,1).contiguous()
            input_shape = z.shape

            # Flatten input
            flat_input = z.view(-1, self.embedding_dim)

            # Calculate distances

            # distances from input to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                        + torch.sum(self.embedding.weight**2, dim=1)
                        - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

            # find closest encodings
            min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            min_encodings = torch.zeros(
                min_encoding_indices.shape[0], self.num_embeddings).cuda() #.to(device)
            min_encodings.scatter(1, min_encoding_indices, 1)


            # get quantized latent vectors
            z_q= torch.matmul(min_encodings, self.embedding.weight).view(input_shape)


            # Loss
            e_latent_loss = F.mse_loss(z_q.detach(), z)
            q_latent_loss = F.mse_loss(z_q, z.detach())
            loss = q_latent_loss + self.commitment_cost * e_latent_loss

            #preserve gradients
            z_q = z + (z_q - z).detach()

            e_mean = torch.mean(min_encodings, dim=0)
            perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

            # convert quantized from BHWC -> BCHW
            return loss, z_q.permute(0, 3, 1, 2).contiguous(), perplexity, min_encodings
