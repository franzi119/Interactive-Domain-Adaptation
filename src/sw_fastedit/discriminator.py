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
          #print('conv1',x.shape)
          x = self.conv2(x)
          #print('conv2', x.shape)
          x = self.conv3(x)
          #print('conv3', x.shape)
          x = self.conv4(x)
          #print('conv4', x.shape)
          x = self.maxpool(x)
          #print('maxpool', x.shape)
          x = x.view(x.size(0), -1)
          x = torch.unsqueeze(x, 0)
          #print('view shape', x.shape)
          x = self.classifier(x)
          #print('classifier', x.shape)
          #print('target:', x)

          return x
     

#https://github.com/wasidennis/AdaptSegNet/blob/master/model/discriminator.py https://arxiv.org/pdf/1802.10349.pdf PADA paper (source 26)
class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		self.conv1 = nn.Conv3d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		self.conv2 = nn.Conv3d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		self.conv3 = nn.Conv3d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		self.conv4 = nn.Conv3d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		self.classifier = nn.Linear(ndf*8, 1)

		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x = self.conv1(x)
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.leaky_relu(x)
		x = self.conv3(x)
		x = self.leaky_relu(x)
		x = self.conv4(x)
		x = self.leaky_relu(x)
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x) 

		return x