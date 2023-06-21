from mesh_to_sdf import mesh_to_voxels
import skimage
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mesh = trimesh.load('./ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj')
# mesh = mesh.dump(concatenate=True)
voxels = mesh_to_voxels(mesh, 64, pad=True)
voxels = truncated_sdf(voxels, 0.4)

# # Create a figure and a 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Use the voxels() method to plot the filled voxels
# ax.voxels(voxels, facecolors='red', edgecolor='k')
#
# # Set the axis limits and labels
# ax.set_xlim(0, voxels.shape[0])
# ax.set_ylim(0, voxels.shape[1])
# ax.set_zlim(0, voxels.shape[2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
#
# # Show the plot
# plt.show()

vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
mesh.show()