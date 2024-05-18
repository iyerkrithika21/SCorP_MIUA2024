import os
import math
import argparse
import json
import torch
import torch.utils.tensorboard
from torch.nn import Module
import pytorch3d
import pytorch3d.loss
import sys

from models.image_branch.encoder import *
from models.mesh_branch.dgcnn import *

from models.imnet import * 

def MSE(predicted, ground_truth):
	return torch.mean((predicted - ground_truth)**2)

class SCorP(Module):
	"""docstring for Baseline3D"""
	def __init__(self, args):
		super(SCorP, self).__init__()
		self.args = args
		self.encoder = ConvolutionalBackbone(args).to(args.device)
		self.dgcnn = DGCNN_AE(args).to(args.device)
		self.imnet = ImNet(in_features=args.emb_dims, nf=args.nf,device=args.device,args=args).to(args.device)
		self.imnet.set_template(args,args.input_x_T.numpy())


	def set_template(self,input_x_T):
		self.input_x_T = input_x_T
		self.imnet.set_template(self.args, self.input_x_T.numpy())

	def update_template(self):
		new_template = self.sample_template()
		new_template = new_template[0,:,:].cpu().detach()
		self.set_template(new_template)
		return new_template

	def sample_template(self, num_estimates = 1000):
		z = torch.randn([num_estimates, self.args.emb_dims]).to(self.args.device)
		
		# mean embedding 
		z_mean = torch.mean(z, axis = 0).reshape((1,-1))
		# mean correspondence using IM-Net
		correspondences_mean = self.imnet(z_mean,self.input_x_T.numpy())
		
		return correspondences_mean

	def sample(self, w, num_points, truncate_std=None):
		batch_size, _ = w.size()
		if truncate_std is not None:
			z = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
		z = w
		correspondences = self.imnet(z,self.input_x_T.detach().numpy())
		return correspondences


	def predict(self, images=None, vertices=None, idx=None):
		if(images != None):
			z = self.encoder(images)
		else:
			z, _ = self.dgcnn(vertices, idx)
		correspondences = self.imnet(z,self.input_x_T.cpu().detach().numpy())
		
		return correspondences

	def get_loss_mesh(self, vertices, label, faces=None, idx=None):

		# mesh flank
		batch_size = vertices.shape[0]
		z_mesh, reconstruction = self.dgcnn(vertices,idx)
		
		m_correspondences = self.imnet(z_mesh, self.input_x_T.detach().numpy())

		if(label==None):
			true_x = vertices
		else:
			true_x = label
		

		if(self.args.mse_weight>0):
			loss_dgcnn = F.mse_loss(true_x.reshape((batch_size,-1,3)), reconstruction.reshape((batch_size,-1,3)), reduction='none')
			loss_dgcnn = loss_dgcnn.mean(axis = (2,1))
			loss_dgcnn =  loss_dgcnn.mean()
		else:
			loss_dgcnn = torch.zeros(1).to(vertices.device)

		if self.args.chamfer_dist == 'L1':
			loss_cd_mesh, _ =  pytorch3d.loss.chamfer_distance(true_x.reshape((batch_size,-1,3)), m_correspondences.reshape((batch_size,-1,3)), point_reduction='mean', batch_reduction='mean', norm=1)
		elif self.args.chamfer_dist == 'L2':
			loss_cd_mesh, _ =  pytorch3d.loss.chamfer_distance(true_x.reshape((batch_size,-1,3)), m_correspondences.reshape((batch_size,-1,3)), point_reduction='mean', batch_reduction='mean', norm=2)
		
		

		

		
		
		loss = loss_cd_mesh  + (self.args.mse_weight*loss_dgcnn)
		return loss, loss_cd_mesh, loss_dgcnn

	def get_loss_image(self, image, vertices, label=None, idx= None):
		# Image flank
		z = self.encoder(image)
		batch_size = vertices.shape[0]
		with torch.no_grad():
			z_mesh, reconstruction = self.dgcnn(vertices,idx)

		latent_loss = F.mse_loss(z, z_mesh, reduction='none')
		latent_loss = latent_loss.sum(axis = (1))
		latent_loss = latent_loss.mean()

		correspondences = self.imnet(z,self.input_x_T.cpu().detach().numpy())
		if(label==None):
			true_x = vertices
		else:
			true_x = label

		if self.args.chamfer_dist == 'L1':
			loss_cd_image, _ =  pytorch3d.loss.chamfer_distance(true_x.reshape((batch_size,-1,3)), correspondences.reshape((batch_size,-1,3)),point_reduction='mean', batch_reduction='mean', norm=1)
			
		elif self.args.chamfer_dist == 'L2':
			loss_cd_image, _ =  pytorch3d.loss.chamfer_distance(true_x.reshape((batch_size,-1,3)), correspondences.reshape((batch_size,-1,3)),point_reduction='mean', batch_reduction='mean', norm=2)

		loss =  latent_loss
		return loss, loss_cd_image, latent_loss
		


	

	def get_loss_imagecd(self, image, vertices, label, faces=None, idx=None):
		# Image flank
		z = self.encoder(image)
		batch_size = vertices.shape[0]
		correspondences = self.imnet(z,self.input_x_T.cpu().detach().numpy())
		with torch.no_grad():
			z_mesh, reconstruction = self.dgcnn(vertices,idx)
		
		latent_loss = F.mse_loss(z, z_mesh, reduction='none')
		latent_loss = latent_loss.sum(axis = (1))
		latent_loss = latent_loss.mean()


		if(label==None):
			true_x = vertices
		else:
			true_x = label
		
		
		if self.args.chamfer_dist == 'L1':
			loss_cd, _ =  pytorch3d.loss.chamfer_distance(true_x.reshape((batch_size,-1,3)), correspondences.reshape((batch_size,-1,3)),point_reduction='mean', batch_reduction='mean', norm=1)  
		elif self.args.chamfer_dist == 'L2':
			loss_cd, _ =  pytorch3d.loss.chamfer_distance(true_x.reshape((batch_size,-1,3)), correspondences.reshape((batch_size,-1,3)),point_reduction='mean', batch_reduction='mean', norm=2)


		loss = latent_loss + loss_cd

		return loss, loss_cd, latent_loss


	def forward(self, images=None, vertices=None, idx=None):
		if(images!=None):
			z = self.encoder(images)
		else:
			z, _ = self.dgcnn(vertices,idx)
			
		correspondences = self.imnet(z,self.input_x_T.cpu().detach().numpy())

		return correspondences