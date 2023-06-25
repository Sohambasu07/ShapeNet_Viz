import numpy as np
import trimesh
import math
from utils import createCmap, viz_trimesh
import pickle

with open('dataset/plane/plane_0.pkl', 'rb') as f:
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



