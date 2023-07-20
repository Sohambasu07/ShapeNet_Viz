import numpy as np
import trimesh
import trimesh.proximity as prox
import math
from utils import*

path = './ShapeNetCore.v2/03211117/111f2a331b626935d82b15a1af434a9f/models/model_normalized.obj'

num_points = 512

mesh = trimesh.load_mesh(path)
meshes = mesh.dump(concatenate=True)
merged_mesh = trimesh.util.concatenate(meshes)
# merged_mesh.show()

gen = np.linspace(-1, 1, math.ceil(num_points**(1/3)))
points3D = np.array([np.array([x, y, z]) for x in gen for y in gen for z in gen])

sdf = prox.signed_distance(merged_mesh, points3D)
# sdf = truncated_sdf(sdf, 0.8)

colormap = createCmap(sdf)

viz_trimesh(merged_mesh, points3D, sdf, colormap)
# viz_matplotlib(merged_mesh, points3D, sdf)