import os
from utils import obj_to_tsdf
import numpy as np
import pickle
import json

root_folder = '../../ShapeNetCore.v2/'
save_root_folder = 'dataset'
if not os.path.exists(save_root_folder):
    os.mkdir(save_root_folder)

class_ids = {'plane': '02691156', 'chair': '03001627', 'table': '04379243'}

num_points = 512
threshold = 0.8
dataset_info = {'num_points':num_points, 'threshold':threshold}
with open(os.path.join(save_root_folder, 'dataset_info.json'), 'w') as fp:
    json.dump(dataset_info, fp)

for cls_name in class_ids:

    class_save_folder = os.path.join(save_root_folder, cls_name)
    if not os.path.exists(class_save_folder):
        os.mkdir(class_save_folder)
    cls_path = os.path.join(root_folder, class_ids[cls_name])
    for i, sample_id in enumerate(os.listdir(cls_path)):
        sample_path = os.path.join(cls_path, sample_id + '/models/model_normalized.obj')
        # try:
        tsdf = obj_to_tsdf(sample_path, num_points, threshold)
        # except FloatingPointError:
        #     print('Cannot convert to sdf')
        #     continue

        tsdf_save_path = os.path.join(class_save_folder, f'{cls_name}_{i}.pkl')
        tsdf_sample = {'tsdf':tsdf, 'model_path':sample_path}

        with open(tsdf_save_path, 'wb') as f:
            pickle.dump(tsdf_sample, f)
        