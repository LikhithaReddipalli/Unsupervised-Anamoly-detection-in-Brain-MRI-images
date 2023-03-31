import torch
import torch.nn as nn
import torch.nn.functional as F



# adapted from https://github.com/dome272/VQGAN-pytorch and https://github.com/CompVis/taming-transformers

class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# residual block with group norm, swish activation and conv layers
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size=3,
                      stride=1, padding=1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size=3,
                      stride=1, padding=1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels= in_channels , out_channels= out_channels,
                                        kernel_size=1, stride=1, padding=0).to(device)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


# upsampling blocks used in decoder for increase the image size to original size
class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


# non-local block
class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A


# encoder block
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.enc_channels = config['enc_channels'] # list of encoder channels
        self.image_channels= config['image_channels'] # number of image channels
        self.attn_resolutions = config['attn_resolutions']
        self.enc_num_res_blocks = config['enc_num_res_blocks'] # number of residual blocks for encoder
        self.enc_resolution = config['enc_resolution'] # encoder resolution
        self.latent_dim = config['latent_dim'] # size of latent vectors
        layers = [nn.Conv2d(self.image_channels, self.enc_channels[0], 3, 1, 1)]
        for i in range(len(self.enc_channels)-1):
            in_channels = self.enc_channels[i]
            out_channels = self.enc_channels[i + 1]
            for j in range(self.enc_num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if self.enc_resolution in self.attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(self.enc_channels)-2:
                layers.append(DownSampleBlock(self.enc_channels[i+1]))
                self.enc_resolution //= 2
        layers.append(ResidualBlock(self.enc_channels[-1], self.enc_channels[-1]))
        layers.append(NonLocalBlock(self.enc_channels[-1]))
        layers.append(ResidualBlock(self.enc_channels[-1], self.enc_channels[-1]))
        layers.append(GroupNorm(self.enc_channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(self.enc_channels[-1], self.latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# decoder block
class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.dec_channels = config['dec_channels'] # list of decoder channels
        self.image_channels = config['image_channels'] # number of image channel
        self.attn_resolutions = config['attn_resolutions']
        self.dec_num_res_blocks = config['dec_num_res_blocks'] # number of residual blocks to be used for decode network
        self.dec_resolution = config['dec_resolution'] # resolution of decoder
        self.latent_dim = config['latent_dim'] # size of latent vectors

        in_channels = self.dec_channels[0]
        layers = [nn.Conv2d(self.latent_dim, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(self.dec_channels)):
            out_channels = self.dec_channels[i]
            for j in range(self.dec_num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if self.dec_resolution in self.attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                self.dec_resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.ConvTranspose2d(in_channels, in_channels, 5, stride=1))
        layers.append(nn.ConvTranspose2d(in_channels, in_channels, 5, stride=1))
        layers.append(nn.ConvTranspose2d(in_channels, in_channels, 5, stride=1))
        layers.append(nn.ConvTranspose2d(in_channels, in_channels, 4, stride=1, padding = 2))
        layers.append(nn.Conv2d(in_channels, self.image_channels, 1, 1, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# similar to code snippet in vqvae for codebook vectors
class Codebook(nn.Module):
    def __init__(self, config):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = config['num_codebook_vectors'] # number of codebook vectors and dimenions can be chnaged in config file for various experiments
        self.latent_dim = config['latent_dim']
        self.beta = config['beta'] # scaling factor for commitment cost

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)


    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z)**2) + self.beta * torch.mean((z_q - z.detach())**2)

        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss


"""
PatchGAN Discriminator (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py#L538)
"""

class Discriminator(nn.Module):
    def __init__(self, config, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()
        self.image_channels = config['image_channels']
        layers = [nn.Conv2d(self.image_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

