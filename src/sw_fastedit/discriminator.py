# based on: https://github.com/wasidennis/AdaptSegNet/blob/master/model/discriminator.py
# Code modification by B.Sc. Franziska Seiz, Karlsuhe Institute of Techonology #
# franzi.seiz96@gmail.com #

from monai.networks.blocks import Convolution
import torch.nn as nn
import torch


class Discriminator(nn.Module):
     def __init__(self, num_in_channels, ndf = 64, pool_size=(4, 4, 4)):
          super(Discriminator, self).__init__()

          self.conv1 = Convolution(spatial_dims=3, in_channels=num_in_channels, out_channels=ndf, act=('leakyrelu'), kernel_size=2, strides=2, padding=1)
          self.conv2 = Convolution(spatial_dims=3, in_channels=ndf, out_channels=ndf*2, act=('leakyrelu'), kernel_size=2, strides=2, padding=1)
          self.conv3 = Convolution(spatial_dims=3, in_channels=ndf*2, out_channels=ndf*4, act=('leakyrelu'), kernel_size=2, strides=2, padding=1)
          self.conv4 = Convolution(spatial_dims=3, in_channels=ndf*4, out_channels=ndf*8, act=('leakyrelu'), kernel_size=2, strides=2, padding=1)
          self.maxpool = nn.AdaptiveMaxPool3d(pool_size)
          self.classifier = nn.Linear(in_features=ndf*8*pool_size[0]*pool_size[1]*pool_size[2],out_features=1)
     
     def forward(self,x):
          x = self.conv1(x)
          x = self.conv2(x)
          x = self.conv3(x)
          x = self.conv4(x)
          x = self.maxpool(x)
          x = x.view(x.size(0), -1)
          x = torch.unsqueeze(x, 0)
          x = self.classifier(x)

          return x
     