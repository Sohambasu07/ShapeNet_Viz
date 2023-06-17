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
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    z_coords = [point[2] for point in points]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], triangles=mesh.faces)
    ax.scatter(x_coords, y_coords, z_coords, c=sdf)
    plt.show()

def viz_trimesh(mesh, points, sdf, colormap):
    scene = trimesh.scene.Scene(mesh)
    point_cloud = trimesh.points.PointCloud(points, colors=colormap)
    # print(point_cloud.shape)
    pt_scene = trimesh.scene.Scene(point_cloud)
    scene.add_geometry(pt_scene)
    scene.show()
    # print(sdf.shape)

def truncated_sdf(sdf, threshold):
    sdf[sdf > threshold] = threshold
    sdf[sdf < -threshold] = -threshold
    return sdf