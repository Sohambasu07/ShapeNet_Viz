import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

def createCmap(sequence, cmap='viridis'):
    cmap = cmx.get_cmap(cmap)
    cNorm = colors.Normalize(vmin=min(sequence), vmax=max(sequence))
    colormap = cmap(cNorm(sequence))
    return colormap

def viz_matplotlib(mesh, points, sdf):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=sdf)
    plt.show()

def viz_trimesh(mesh, points, sdf, colormap):
    scene = trimesh.scene.Scene(mesh)
    point_cloud = trimesh.points.PointCloud(points, colors=colormap)
    # print(point_cloud.shape)
    scene.add_geometry(point_cloud)
    scene.show()
    # print(sdf.shape)

def truncated_sdf(sdf, threshold):
    sdf[sdf > threshold] = threshold
    sdf[sdf < -threshold] = -threshold
    return sdf