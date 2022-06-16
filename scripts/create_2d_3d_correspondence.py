import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from scripts.compute_iou import create_line_mesh, create_line_mesh_from_dict
from utils.directory import fetch_split_train_path, fetch_split_test_path, fetch_object_2d_3d_path, \
    fetch_xyzrgb_pcd_path, fetch_xyzrgb_bbox_path, fetch_object_pair_path, fetch_xyzrgb_mask_path, fetch_segformer_path
from utils.intrinsic_fetcher import IntrinsicFetcher
from utils.meta_io import MetaObject2DV2


def build_single_meta_file():
    meta_dict_by_image_id = torch.load('/home/junha/data/sunrefer/meta.pt')
    image_id = '000001'
    meta_dict = meta_dict_by_image_id[image_id]
    print(meta_dict.keys())

    import open3d as o3d
    pcd_np = np.load(fetch_xyzrgb_pcd_path(image_id))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.reshape(pcd_np[..., :3], (-1, 3)))

    from scipy.spatial.transform import Rotation as R
    r = R.from_euler('y', 90, degrees=True)

    t = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    r_ = np.eye(4, dtype=np.float32)
    r_[:3, :3] = r.as_matrix()

    objects = [pcd]
    for obj_3d in meta_dict['obj_3d']:
        print(obj_3d)  # class_name, basis, coeffs, centroid, orientation
        line_mesh = create_line_mesh_from_dict(bbox_dict=obj_3d, aabb=False)
        line_mesh.transform(t)
        line_mesh.transform(r_)
        objects += line_mesh.cylinder_segments
    o3d.visualization.draw_geometries(objects)


    # create_line_mesh()

    return

    # scene2d_by_image_id = fetch_scene_object_by_image_id('v2_2d')
    # torch.save(scene2d_by_image_id, '/home/junha/Downloads/scene2d.pt')
    # scene3d_by_image_id = fetch_scene_object_by_image_id('v2_3d')
    # torch.save(scene3d_by_image_id, '/home/junha/Downloads/scene3d.pt')
    # return

    scene2d_by_image_id: Dict[str, MetaObject2DV2] = torch.load('/home/junha/Downloads/scene2d.pt')
    bbox3d_by_image_id: Dict[str, List[Tuple[str, List[float]]]] = json.load(open(str(fetch_xyzrgb_bbox_path()), 'r'))
    res = dict()
    for image_id, scene2d in list(scene2d_by_image_id.items()):
        bbox3d_list = bbox3d_by_image_id[image_id]
        class_dict = scene2d.build_class_dict()

        object_2d_and_3d_list = []
        examine_3d_list = []
        for i, bbox3d in enumerate(bbox3d_list):
            class_name, coordinates_3d = bbox3d
            if class_name not in class_dict:
                print('{}::invalid 3d bbox: {}, {}'.format(image_id, i, class_name))
                # print(list(map(lambda x: x.class_name, scene2d.gt_2d_bbox)))
                continue
            if len(class_dict[class_name]) > 1:
                examine_3d_list.append((i, bbox3d))
            else:
                bbox2d = class_dict[class_name][0]
                object_2d_and_3d_list.append((class_name, bbox2d.object_id, bbox2d.gt_bbox_2d, i, coordinates_3d))

        if examine_3d_list:
            xyzrgb_pcd = np.load(str(fetch_xyzrgb_pcd_path(image_id)))
            for i, bbox3d in examine_3d_list:
                class_name, coordinates_3d = bbox3d
                bbox2d_list = class_dict[class_name]
                dist_bbox_list = []
                for bbox2d in bbox2d_list:
                    x, y, w, h = bbox2d.gt_bbox_2d
                    xyz = xyzrgb_pcd[y:y + h, x:x + w, :3]
                    mask = np.logical_and(
                        np.logical_and(np.abs(xyz[:, :, 0]) < 1e-5,
                                       np.abs(xyz[:, :, 1]) < 1e-5),
                        np.abs(xyz[:, :, 2]) < 1e-5)
                    effective_xyz = xyz[~mask, :]
                    num_points = effective_xyz.shape[0]
                    if num_points < 10:
                        continue
                    sampled_xyz = effective_xyz[np.random.choice(num_points, 1000), :]
                    centroid = np.mean(sampled_xyz, axis=0)
                    dist = np.linalg.norm(centroid - coordinates_3d[:3])
                    dist_bbox_list.append((dist, bbox2d))
                dist_bbox_list = sorted(dist_bbox_list, key=lambda x: x[0])
                if not dist_bbox_list:
                    continue
                dist, bbox2d = dist_bbox_list[0]
                if dist > 1.:
                    print('{}::too far from the gt center: {}'.format(image_id, dist))
                else:
                    object_2d_and_3d_list.append((class_name, bbox2d.object_id, bbox2d.gt_bbox_2d, i, coordinates_3d))

        object_2d_and_3d_list = sorted(object_2d_and_3d_list, key=lambda x: x[1])
        res[image_id] = object_2d_and_3d_list

    with open('/home/junha/Downloads/object_2d_3d.json', 'w') as file:
        json.dump(res, file, indent=4)


def verify_single_meta_file():
    with open('/home/junha/Downloads/object_2d_3d.json', 'r') as file:
        object_2d_3d = json.load(file)
    for image_id, items in tqdm(list(object_2d_3d.items())):
        xyz = np.load(str(fetch_xyzrgb_pcd_path(image_id)))
        for class_name, obj_id_2d, bbox_2d, obj_id_3d, bbox_3d in items:
            x, y, w, h = bbox_2d
            mask = np.logical_or(np.logical_or(np.abs(xyz[:, :, 0]) > 1e-5, np.abs(xyz[:, :, 1]) > 1e-5),
                                 np.abs(xyz[:, :, 2]) > 1e-5)
            mask_crop = mask[y:y + h, x:x + w]
            dx = xyz[y:y + h, x:x + w, 3][mask_crop] * xyz.shape[1]
            dy = xyz[y:y + h, x:x + w, 4][mask_crop] * xyz.shape[0]
            in_dx = np.logical_and(dx >= x, dx <= x + w)
            in_dy = np.logical_and(dy >= y, dy <= y + h)
            in_dxdy = np.logical_and(in_dx, in_dy)
            inlier_rate = np.sum(in_dxdy) / in_dx.shape[0] * 100
            if inlier_rate < 99.:
                print(image_id, class_name, 'inlier rate: {:5.3f}'.format(inlier_rate))


if __name__ == '__main__':
    build_single_meta_file()
