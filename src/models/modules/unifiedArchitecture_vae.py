import torch
import numpy as np
import math
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import functools


### Unified Network architecture Encoder 2D/3D ###
class Encoder_unified(nn.Module):  # Unified Encoder from Baur et al.
    def __init__(self, config):
        super(Encoder_unified, self).__init__()

        self.imageDim = config['imageDim']  # Input Dimension of the Image
        self.channelsEnc = config['fmapsEnc']  # Number of Featuremaps for each Layer
        self.channelsDec = config['fmapsDec']  # Number of Featuremaps for each Layer
        self.kernelSize = config['kernelSize']  # KernelSize
        self.intermediateResolutions = config['interRes']
        self.activation = nn.LeakyReLU()
        self.spatialDim = config['spatialDims']
        self.factor = config['rescaleFactor']
        ### adapted implantation  from Baur ### very flexible when it comes to different layer choices and imagedimensions!

        # calculate the number of necessary pooling operations (2d and 3D)
        numPooling = math.ceil(math.log(config['imageDim'][0], 2) - math.log(float(self.intermediateResolutions[0]), 2))

        if self.spatialDim == '2D':
            numPooling_depth = 0
        maxnumPooling = np.max([numPooling_depth, numPooling])


        En = []
        for i in range(maxnumPooling):
            numFmap = int(min(self.channelsEnc[-1], self.channelsDec[-1] * (2 ** i)))

            if i == 0:
                numFmapHist = self.channelsEnc[0]

            if self.spatialDim == '2D':
                En.append(nn.Conv2d(in_channels=numFmapHist,
                                    out_channels=numFmap,
                                    kernel_size=self.kernelSize,
                                    stride=2,
                                    padding=2,
                                    bias=False))

            numFmapHist = numFmap  # update last numFmap

            if self.spatialDim == '2D':
                En.append(nn.BatchNorm2d(numFmap))

            En.append(self.activation)

        self.Enc = nn.Sequential(*En)

    def forward(self, x):
        x = self.Enc(x)
        return x


### Unified Network architecture Decoder ###
class Decoder_unified(nn.Module):  # Unified Decoder from Baur et al.
    def __init__(self, config):
        super(Decoder_unified, self).__init__()

        self.imageDim = config['imageDim']  # Input Dimension of the Image
        self.channelsDec = config['fmapsDec']  # Number of Featuremaps for each Layer
        self.channelsEnc = config['fmapsEnc']  # Number of Featuremaps for each Layer
        self.kernelSize = config['kernelSize']  # KernelSize
        self.intermediateResolutions = config['interRes']
        self.interpolation = config['cropMode']
        self.factor = config['rescaleFactor']
        self.activation = nn.LeakyReLU()
        self.spatialDim = config['spatialDims']
        ### adapted implantation  from Baur ### very flexible when it comes to different layer choices and im agedimensions!
        # calculate the number of necessary pooling operations (2d and 3D)
        numUpsampling = math.ceil(
            math.log(config['imageDim'][0], 2) - math.log(float(self.intermediateResolutions[0]), 2))

        if self.spatialDim == '2D':
            numUpsampling_depth = 0
        maxnumUpsampling = np.max([numUpsampling_depth, numUpsampling])

        De = []
        for i in range(maxnumUpsampling):
            numFmap = int(max(self.channelsDec[-1], self.channelsEnc[-1] / (2 ** i)))
            #numFmap = numFmap/2

            if i == 0:
                numFmapHist = self.channelsDec[0]

            if self.spatialDim == '2D':

                if i == 0:
                    De.append(nn.ConvTranspose2d(in_channels=numFmapHist,
                                             out_channels=numFmap,
                                             kernel_size=self.kernelSize,
                                             stride=2,
                                             padding=2,
                                             bias=False))

                if i == 1:
                    De.append(nn.ConvTranspose2d(in_channels=numFmapHist,
                                                 out_channels=numFmap,
                                                 kernel_size=self.kernelSize,
                                                 stride=2,
                                                 padding=1,
                                                 bias=False))

                if i == 2:
                    De.append(nn.ConvTranspose2d(in_channels=numFmapHist,
                                                 out_channels=numFmap,
                                                 kernel_size=self.kernelSize,
                                                 stride=2,
                                                 padding=1,
                                                 bias=False))
                if i== 3:
                    De.append(nn.ConvTranspose2d(in_channels=numFmapHist,
                                                out_channels=numFmap,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                bias=False))

                '''else:
                    De.append(nn.ConvTranspose2d(in_channels=numFmapHist,
                                                out_channels=numFmap,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0,
                                                bias=False))'''


                numFmapHist = numFmap  # update last numFmap

            if self.spatialDim == '2D':
                De.append(nn.BatchNorm2d(num_features=numFmap))
                De.append(self.activation)
        self.Dec = nn.Sequential(*De)

    def forward(self, x):

        x = self.Dec(x)
        return x
