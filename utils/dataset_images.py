import torch
import os
import sys
import glob
import pickle
import numpy as np
from torch.utils.data import Dataset
import nrrd
import pyvista as pv
'''
Reads .nrrd files and returns numpy array
Whitens/normalizes images
'''
def get_images(image_dir, image_files):
	
	all_images = []
	for image_file in image_files:
		img, header = nrrd.read(image_file)
		all_images.append(img)
		
	all_images = np.array(all_images)
	# get mean and std
	mean_path = image_dir + 'mean_img.npy'
	std_path = image_dir + 'std_img.npy'
	if not os.path.exists(mean_path) or not os.path.exists(std_path):
		mean_image = np.mean(all_images)
		std_image = np.std(all_images)
		np.save(mean_path, mean_image)
		np.save(std_path, std_image)
	else:
		mean_image = np.load(mean_path)
		std_image = np.load(std_path)
	# normalize
	norm_images = []
	for image in all_images:
		norm_images.append([(image-mean_image)/std_image])
	return np.array(norm_images), image_files

def get_image_particle_pairs(image_dir, particle_dir, mesh_dir, image_files, mesh_extension):
	all_images = []
	all_particles = []
	all_vertices = []
	max_size = 0
	for image_file in image_files:
		img, header = nrrd.read(image_file)
		all_images.append(img)

		# read corresponding particle
		particle_filename = os.path.basename(image_file).replace(".nrrd", "_world.particles")
		mesh_name = f'{mesh_dir}/{os.path.basename(image_file).replace(".nrrd", "."+mesh_extension)}'
		mesh = pv.read(mesh_name)
		vertices = np.array(mesh.points)[:5000,:]
		p = np.loadtxt(f'{particle_dir}/{particle_filename}')
		
		all_particles.append(p)
		all_vertices.append(vertices)
		if (len(vertices)>max_size):
			max_size = len(vertices)
	all_images = np.array(all_images)
	all_vertices = np.array(all_vertices)
	all_particles = np.array(all_particles)

	

	# get mean and std
	mean_path = image_dir + 'mean_img.npy'
	std_path = image_dir + 'std_img.npy'
	if not os.path.exists(mean_path) or not os.path.exists(std_path):
		mean_image = np.mean(all_images)
		std_image = np.std(all_images)
		np.save(mean_path, mean_image)
		np.save(std_path, std_image)
	else:
		mean_image = np.load(mean_path)
		std_image = np.load(std_path)
	# normalize
	norm_images = []
	for image in all_images:
		norm_images.append([(image-mean_image)/std_image])
	return np.array(norm_images), image_files, all_particles, all_vertices



class Imagedataset(Dataset):
	def __init__(self, image_dir,extention='.nrrd'):
		# get all images
		image_files = glob.glob(f'{image_dir}/*{extention}')
		self.input_images, filenames = get_images(image_dir, image_files)
		self.input_images = torch.FloatTensor(self.input_images)
		self.names = [os.path.basename(filename).split(".")[0] for filename in filenames]

	def __getitem__(self, index):
		x = self.input_images[index]
		name = self.names[index]
		return x, name
	def __len__(self):
		return len(self.names)


class DeepSSMImageDataset(Dataset):
	def __init__(self, args,partition):
		# get all images
		
		self.image_dir = args.data_dir + f'/{partition}/images/'
		self.particle_dir = args.data_dir + f'/{partition}/particles/'
		self.mesh_dir = args.data_dir + f'/{partition}/meshes/'
		image_files = sorted(glob.glob(f'{self.image_dir}/*.{args.image_extension}'))

		self.input_images, filenames, self.particles, self.vertices = get_image_particle_pairs(self.image_dir, self.particle_dir, self.mesh_dir, image_files, args.mesh_extension)
		self.input_images = torch.FloatTensor(self.input_images)
		self.particles = torch.FloatTensor(self.particles)
		self.vertices = self.vertices
		self.names = [os.path.basename(filename).split(".")[0] for filename in filenames]

	def __getitem__(self, index):
		x = self.input_images[index]
		p = self.particles[index]
		name = self.names[index]
		v = self.vertices[index]
		data_dict = {
		'image': x,
		'name': name,
		'particles':p,
		'vertices': v
		}
		return data_dict

	def __len__(self):
		return len(self.names)