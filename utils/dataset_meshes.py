import os
import sys
import glob
import pickle
import numpy as np
from torch.utils.data import Dataset
import pyvista as pv
from torch_geometric.utils import geodesic_distance
import torch
from scipy.spatial.distance import pdist, squareform
import time



def geodescis(pos, face,k, max_gdist):
    
    pos= torch.Tensor(pos)
    face = torch.Tensor(face)
    dist = -1*geodesic_distance(pos,face.t(),norm=False,num_workers=-1, max_distance = max_gdist)
    idx = dist.topk(k=k,dim=-1)[1]
    return idx

def load_meshes_with_faces(directory, partition, extention,k, max_gdist=None):

    if(max_gdist==None):
        max_gdist = 5
    
    files = sorted(glob.glob(directory + "/*."+extention))
     
    if(len(files) == 0):
        raise Exception("No files. Check the extention of meshes specified") 
    max_size = 0
    vertices_all = []
    faces_all = []
    pk_filename = directory + 'idx_' + str(k) + "_" +  partition + f'_geodescis_{max_gdist}.pkl'
    try:
        save = False
        with open(pk_filename, 'rb') as f:
            idx_all = pickle.load(f)

    except:
        save = True
        idx_all = {}
    
    max_scale = 0
    filename = []
    for f in files:
        name = f.split("/")[-1]
        filename.append(name)
        
        mesh = pv.read(f)
        vertices = np.array(mesh.points).astype('float')
        faces = np.asarray(mesh.faces).reshape((-1, 4))[:, 1:]
        if (save == True):
            idx = geodescis(vertices, faces,k, max_gdist)
            idx_all[name] = idx[:,:k]
            with open(pk_filename, 'wb') as f:
                pickle.dump(idx_all,f)
        scale = np.max(np.abs(vertices))
        if(scale>max_scale):
            max_scale = scale
        vertices_all.append(vertices)
        faces_all.append(faces)
        
        if (len(vertices)>max_size):
            max_size = len(vertices)
        
    if(save ==True):
        with open(pk_filename, 'wb') as f:
            pickle.dump(idx_all,f)
    
    return vertices_all, faces_all,idx_all, max_size, max_scale, filename


class MeshesWithFaces(Dataset):
    def __init__(self, args, scale_mode, partition = 'train', extention = "ply", k = 10, noise_level=0,shuffle_points=False,size=None):
        self.mesh_dir = args.data_dir + f'/{partition}/meshes/'
        self.data, self.faces_all, self.idx_all, self.max_size, self.scale, self.filename = load_meshes_with_faces(self.mesh_dir, partition, args.mesh_extension,args.k,args.max_gdist)
        self.partition = partition     
        self.noise_level = noise_level
        self.scale_mode = scale_mode
        self.shuffle_points = shuffle_points
        if(size!=None):
            self.max_size = size


    def __getitem__(self, item):
        name = self.filename[item]
        pointcloud = self.data[item]
        faces = self.faces_all[item]

        excess = self.max_size - len(pointcloud)
        idx = self.idx_all[name]
        idx_extended = idx
        

        list_idx = list(range(len(pointcloud)))
        if(excess > 0):
            repeat_idx = np.random.randint(0,len(pointcloud),excess)
            list_idx = list_idx + list(repeat_idx)

        pc = pointcloud[list_idx,:]
        idx_extended = idx[list_idx,:]
        faces = faces[list_idx, :]
        
        if(self.shuffle_points):
            s_idx = np.random.randint(0,len(pc),len(pc)).tolist()
            pc = pointcloud[s_idx,:]
            idx_extended = idx[s_idx,:]
            faces = faces[s_idx, :]
        
        pc = torch.from_numpy(pc).type(torch.float)
        if self.scale_mode == 'global_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = self.stats['std'].reshape(1, 1)
        elif self.scale_mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif self.scale_mode == 'shape_half':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.5)
        elif self.scale_mode == 'shape_34':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.75)
        elif self.scale_mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        else:
            shift = torch.zeros([1, 3])
            scale = torch.ones([1, 1])

        pc = (pc - shift) / scale
        max_value = pc.max()
        
        if(self.noise_level>0):
            noisy_pc = pc + ((self.noise_level*max_value) * np.random.randn(*pc.shape))#.to_numpy()
        else:
            noisy_pc = pc

        noisy_pc = noisy_pc.type(torch.float)
        pc = pc.type(torch.float)
        
        pc_dict = {
                    'pointcloud': noisy_pc,
                    'name':name,
                    'shift': shift,
                    'scale': scale,
                    'idx': idx_extended,
                    'true_pointcloud': pc,
                    'faces': faces
                }
        return pc_dict
        
    def __len__(self):
        return len(self.data)
