import torch
import torch.nn as nn
from src.models.modules.vqgan_architechture import Encoder, Decoder,Codebook


class vqgan_model_2d(nn.Module):
    def __init__(self, cfg):
        super(vqgan_model_2d, self).__init__()
        self.enc_channels = cfg.enc_channels # list of encoder channels
        self.enc_num_res_blocks = cfg.enc_num_res_blocks # number of residual blocks for encoder network
        self.enc_resolution = cfg.enc_resolution
        self.dec_channels = cfg.dec_channels  # list of decoder channels
        self.dec_num_res_blocks = cfg.dec_num_res_blocks  # number of residual blocks for decoder network
        self.dec_resolution = cfg.dec_resolution
        self.image_channels = cfg.image_channels
        self.latent_dim = cfg.latent_dim # size of latent vectors
        self.attn_resolutions = cfg.attn_resolutions
        self.num_codebook_vectors = cfg.num_codebook_vectors # number of codebook vectors
        self.beta = cfg.beta
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.codebook = Codebook(cfg)
        self.quant_conv = nn.Conv2d(self.latent_dim, self.latent_dim, 1)
        self.post_quant_conv = nn.Conv2d(self.latent_dim, self.latent_dim, 1)

#function to get reconstructed images and vq_loss
    def forward(self, inputs):
        encoded_images = self.encoder(inputs)
        #encoded_images.requires_grad = True
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, vq_loss = self.codebook(quant_conv_encoded_images)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        x_recon = self.decoder(post_quant_conv_mapping)

        outputs = {}
        outputs['vq_loss'] = vq_loss
        outputs['x_recon'] = x_recon
        outputs['codebook_indices'] = codebook_indices
        outputs['codebook_mapping'] = codebook_mapping

        return outputs

    # function to get codebook mappings
    def encode(self, inputs):
        encoded_images = self.encoder(inputs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

#function returns lambda value that is to be used to sclae discriminator loss
    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True, allow_unused= False)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=False)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    def adopt_weight(self,disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor