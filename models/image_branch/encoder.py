import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

def poolOutDim(inDim, kernel_size, padding=0, stride=0, dilation=1):
	if stride == 0:
		stride = kernel_size
	num = inDim + 2*padding - dilation*(kernel_size - 1) - 1
	outDim = int(np.floor(num/stride + 1))
	return outDim
class ConvolutionalBackbone(nn.Module):
	def __init__(self, args):
		super(ConvolutionalBackbone, self).__init__()
		
		# basically using the number of dims and the number of poolings to be used 
		# figure out the size of the last fc layer so that this network is general to 
		# any images

		self.out_fc_dim = np.copy(args.img_dims)
		self.num_latent = args.emb_dims
		padvals = [4, 8, 8]
		for i in range(3):
			self.out_fc_dim[0] = poolOutDim(self.out_fc_dim[0] - padvals[i], 2)
			self.out_fc_dim[1] = poolOutDim(self.out_fc_dim[1] - padvals[i], 2)
			self.out_fc_dim[2] = poolOutDim(self.out_fc_dim[2] - padvals[i], 2)
		
		self.conv = nn.Sequential(OrderedDict([
			('conv1', nn.Conv3d(1, 12, 5)),
			('bn1', nn.BatchNorm3d(12)),
			('relu1', nn.PReLU()),
			('mp1', nn.MaxPool3d(2)),

			('conv2', nn.Conv3d(12, 24, 5)),
			('bn2', nn.BatchNorm3d(24)),
			('relu2', nn.PReLU()),
			('conv3', nn.Conv3d(24, 48, 5)),
			('bn3', nn.BatchNorm3d(48)),
			('relu3', nn.PReLU()),
			('mp2', nn.MaxPool3d(2)),

			('conv4', nn.Conv3d(48, 96, 5)),
			('bn4', nn.BatchNorm3d(96)),
			('relu4', nn.PReLU()),
			('conv5', nn.Conv3d(96, 192, 5)),
			('bn5', nn.BatchNorm3d(192)),
			('relu5', nn.PReLU()),
			('mp3', nn.MaxPool3d(2)),
		]))
		# input(self.out_fc_dim)
		self.fc = nn.Sequential(OrderedDict([
			('flatten', Flatten()),
			('fc1', nn.Linear(self.out_fc_dim[0]*self.out_fc_dim[1]*self.out_fc_dim[2]*192, 384)),
			('relu6', nn.PReLU()),
			('fc2', nn.Linear(384, 96)),
			('relu7', nn.PReLU()),
			('fc3', nn.Linear(96, self.num_latent))
		]))

	def forward(self, x):
		x_conv_features = self.conv(x)
		x_features = self.fc(x_conv_features)
		return x_features



class DeterministicLinearDecoder(nn.Module):
	def __init__(self, args):
		super(DeterministicLinearDecoder, self).__init__()
		self.args = args
		self.num_latent = args.emb_dims
		self.numL = args.num_corr
		self.fc_fine = nn.Linear(self.num_latent, self.numL*3)

	def forward(self, x):
		corr_out = self.fc_fine(x).reshape(-1, self.numL, 3)
		return corr_out