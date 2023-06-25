import numpy as np
import trimesh
import math
from utils import createCmap, viz_trimesh
import pickle
import os

from torch.utils.data import Dataset

class ShapeNet(Dataset):
    def __init__(self, datase_dir):
        self.paths = []
        for class_folder in os.listdir(datase_dir):
            class_dir = os.path.join(datase_dir, class_folder)
            for tsdf_sample in os.listdir(class_dir):
                sample_path = os.path.join(class_dir, tsdf_sample)
                self.paths.append(sample_path)

    
    def __getitem__(self, index):
        with open(self.paths[index], 'rb') as f:
            tsdf = pickle.load(f)
        return tsdf['tsdf']
    
    def __len__(self):
        return len(self.paths)

if __name__ == '__main__':

    with open('dataset/plane/plane_3.pkl', 'rb') as f:
        tsdf_sample = pickle.load(f)


    tsdf= tsdf_sample['tsdf']
    model_path = tsdf_sample['model_path']

    num_points = 512

    mesh = trimesh.load_mesh(model_path)
    meshes = mesh.dump(concatenate=True)
    merged_mesh = trimesh.util.concatenate(meshes)
    # merged_mesh.show()

    grid = np.linspace(-1, 1, math.ceil(num_points**(1/3)))
    points3D = np.array([np.array([x, y, z]) for x in grid for y in grid for z in grid])

    colormap = createCmap(tsdf)

    viz_trimesh(merged_mesh, points3D, tsdf, colormap)



