import numpy as np
import trimesh
import trimesh.proximity as prox
import math
from utils import*
import os
import pickle
import argparse

def visualize_mesh(path, num_points=512):
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


def visualize_tsdf(path, check=True):
    tsdf, _ = obj_to_tsdf(path, 0.8, 64)
    display_tsdf(tsdf)
    if check and not check_voxels(tsdf):
        print("Invalid tsdf")

def check_tsdfs(root_folder):
    invalid = 0
    total = 0
    with open('./invalid_tsdfs.txt', 'w') as invalid_tsdfs:
        lists = os.listdir(root_folder)
        classes = [l for l in lists if os.path.isdir(os.path.join(root_folder, l))]
        for cls_name in classes:
            cls_path = os.path.join(root_folder, cls_name)
            for tsdf_name in os.listdir(cls_path):
                tsdf_path = os.path.join(cls_path, tsdf_name)
                with open(tsdf_path, 'rb') as f:
                    tsdf_sample = pickle.load(f)
                    tsdf = tsdf_sample['tsdf']
                    total += 1
                    if not check_voxels(tsdf):
                        invalid += 1
                        invalid_tsdfs.write(f"{tsdf_path}\n")
                        print({tsdf_path})
    print(f"Total tsdfs: {total}")
    print(f"Total invalid tsdfs: {invalid}")
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, default='./ShapeNetCore.v2')
    parser.add_argument('--sub_folder', type=str, default='03211117')
    parser.add_argument('--model_id', type=str, default='111f2a331b626935d82b15a1af434a9f')
    parser.add_argument('--num_points', type=int, default=512)
    parser.add_argument('--function', type=str, default='viz')

    args = parser.parse_args()

    root_folder = args.root_folder
    sub_folder = args.sub_folder
    model_id = args.model_id
    num_points = args.num_points
    function = args.function

    path = os.path.join(root_folder, sub_folder, model_id, 'models/model_normalized.obj')

    if function == 'viz':
        visualize_mesh(path, num_points)
    elif function == 'tsdf':
        visualize_tsdf(path)
    elif function == 'check':
        check_tsdfs(root_folder)
    