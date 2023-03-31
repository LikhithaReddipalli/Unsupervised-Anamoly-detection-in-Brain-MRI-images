import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms


# loss function to calculate reconstruction loss and combined loss
class L2_vqvae(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,outputs,inputs):

        data_recon = outputs['x_recon']
        vq_loss = outputs['vq_loss']
        data_variance = torch.var(inputs)

        recon_loss1 = nn.MSELoss()
        recon_loss11 = recon_loss1(data_recon, inputs)
        combined_loss = recon_loss11 + vq_loss

        loss = {}
        loss['combined_loss'] = combined_loss
        loss['recon_error'] = recon_loss11
        loss['vq_loss'] = vq_loss

        return loss



