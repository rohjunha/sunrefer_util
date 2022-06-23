import json
from collections import defaultdict
from itertools import product

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

from scripts.compute_iou import create_line_mesh_from_center_size
from utils.directory import fetch_concise_refer_path, fetch_concise_correspondence_path, fetch_concise_data_dir
from utils.pcd_handler import PointCloudHandler


def examine_pcd_inliers(pcd: np.array, aabb: np.array) -> np.array:
    p1 = aabb[:3] - 0.5 * aabb[3:]
    p2 = aabb[:3] + 0.5 * aabb[3:]
    return np.logical_and(*[(p1[d] <= pcd[:, d]) & (pcd[:, d] <= p2[d]) for d in range(3)])


def create_correspondences():
    inlier_threshold = 0.2

    pcd_handler = PointCloudHandler()
    image_id_list = ['{:06d}'.format(i) for i in range(1, 10336)]
    correspondence_dict = dict()
    for image_id in tqdm(image_id_list):
        scene = pcd_handler.scene_dict[image_id]

        # raw_color = pcd_handler.fetch_raw_rgb(image_id)
        # pcd_, mask_, _, _ = pcd_handler.fetch_pcd(image_id)
        # pcd_[~mask_, :] = 0.
        # o3d_pcd = o3d.geometry.PointCloud()
        # o3d_pcd.points = o3d.utility.Vector3dVector(pcd_[mask_, :])
        # o3d_pcd.colors = o3d.utility.Vector3dVector(raw_color[mask_, :].astype(np.float64) / 255.)

        # o3d_obj_list = [o3d_pcd]
        # for obj_3d in scene.obj_3d_list:
        #     line_mesh = create_line_mesh_from_center_size(obj_3d.aabb)
        #     o3d_obj_list += line_mesh.cylinder_segments
        #
        # for obj_2d in scene.obj_2d_list:
        #     x1, y1, w, h = obj_2d.bbox
        #     cv2.rectangle(raw_color, (x1, y1), (x1 + w, y1 + h), color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        # cv2.imwrite('/home/junha/Downloads/{}_orig.jpg'.format(image_id), raw_color)
        #
        # # o3d.visualization.draw_geometries(o3d_obj_list)

        correspondences = []

        count_2d_dict = defaultdict(list)
        count_3d_dict = defaultdict(list)
        for i, obj_2d in enumerate(scene.obj_2d_list):
            count_2d_dict[obj_2d.class_name].append(i)
        for i, obj_3d in enumerate(scene.obj_3d_list):
            count_3d_dict[obj_3d.class_name].append(i)

        class_name_set = set(count_2d_dict.keys()).intersection(set(count_3d_dict.keys()))
        for class_name in class_name_set:
            index_2d_list = count_2d_dict[class_name]
            index_3d_list = count_3d_dict[class_name]

            if len(index_2d_list) == 1 and len(index_3d_list) == 1:
                i, j = index_2d_list[0], index_3d_list[0]
                obj_2d, obj_3d = scene.obj_2d_list[i], scene.obj_3d_list[j]
                pcd, mask, _, _ = pcd_handler.fetch_cropped_pcd(image_id, obj_2d.bbox)
                eff_pcd = pcd[mask, :]
                inliers = examine_pcd_inliers(eff_pcd, obj_3d.aabb)
                inlier_ratio = np.sum(inliers) / eff_pcd.shape[0]
                if inlier_ratio > inlier_threshold:
                    correspondences.append((i, j, class_name, inlier_ratio))

            elif index_2d_list and index_3d_list:
                inlier_ratio_list = []
                for i in index_2d_list:
                    obj_2d = scene.obj_2d_list[i]
                    pcd, mask, _, _ = pcd_handler.fetch_cropped_pcd(image_id, obj_2d.bbox)
                    eff_pcd = pcd[mask, :]
                    for j in index_3d_list:
                        obj_3d = scene.obj_3d_list[j]
                        inliers = examine_pcd_inliers(eff_pcd, obj_3d.aabb)
                        inlier_ratio = np.sum(inliers) / eff_pcd.shape[0]
                        inlier_ratio_list.append((i, j, inlier_ratio))

                inlier_ratio_list = sorted(inlier_ratio_list, key=lambda x: x[-1], reverse=True)
                i, j, inlier_ratio = inlier_ratio_list[0]
                if inlier_ratio > inlier_threshold:
                    correspondences.append((i, j, class_name, inlier_ratio))

        if correspondences:
            correspondence_dict[image_id] = correspondences

        # print(correspondences)
        #
        # return
        #
        # for i, obj_2d in enumerate(scene.obj_2d_list):
        #     pcd, mask, _, _ = pcd_handler.fetch_cropped_pcd(image_id, obj_2d.bbox)
        #     eff_pcd = pcd[mask, :]
        #
        #     if obj_2d.has_3d:
        #         if len(scene.obj_3d_list) > i:
        #             obj_3d = scene.obj_3d_list[i]
        #             inliers = examine_pcd_inliers(eff_pcd, obj_3d.aabb)
        #             inlier_ratio = np.sum(inliers) / eff_pcd.shape[0]
        #
        #             if obj_3d.class_name == obj_2d.class_name and inlier_ratio > inlier_threshold:
        #                 correspondences.append((i, i, obj_2d.class_name, inlier_ratio))
        #             else:
        #                 uncertain_correspondences.append(
        #                     (i, i, obj_2d.has_3d, obj_2d.class_name, obj_3d.class_name, inlier_ratio))
        #     else:
        #         inlier_ratio_list = []
        #         for j in count_3d_dict[obj_2d.class_name]:
        #             obj_3d = scene.obj_3d_list[j]
        #             assert obj_3d.class_name == obj_2d.class_name
        #             inliers = examine_pcd_inliers(eff_pcd, obj_3d.aabb)
        #             inlier_ratio = np.sum(inliers) / eff_pcd.shape[0]
        #             inlier_ratio_list.append((j, inlier_ratio))
        #         print(inlier_ratio_list)
        #             # if inlier_ratio > inlier_threshold:
        #             #     certain_correspondences.append(
        #             #         (i, j, obj_2d.has_3d, obj_2d.class_name, obj_3d.class_name, inlier_ratio))
        #
        # for k, v in count_2d_dict.items():
        #     print(k, v)
        # for k, v in count_3d_dict.items():
        #     print(k, v)
        # print()
        #
        # print(correspondences)
        # print(uncertain_correspondences)

    with open('/home/junha/data/sunrefer/xyzrgb_concise/correspondence.json', 'w') as file:
        json.dump(correspondence_dict, file)


def copy_correspondence():
    old_corr_dict = json.load(open('/home/junha/data/sunrefer/object_2d_3d.json', 'r'))
    new_corr_dict = dict()
    for image_id, corr_items in old_corr_dict.items():
        new_corr_dict[image_id] = sorted([(i, j, cls_name) for cls_name, i, bbox2d, j, bbox3d in corr_items])
    json.dump(new_corr_dict, open(str(fetch_concise_data_dir() / 'correspondence.json'), 'w'))


def verify_correspondence():
    corr_dict = json.load(open(str(fetch_concise_correspondence_path()), 'r'))
    refer_dict = json.load(open(str(fetch_concise_refer_path()), 'r'))
    for anno_id, refer_item in tqdm(list(refer_dict.items())):
        image_id, object_id = anno_id.split('_')
        object_id = int(object_id)
        if image_id not in corr_dict:
            print(anno_id)
        else:
            corr_items = sorted(corr_dict[image_id])
            indices = list(map(lambda x: x[0], corr_items))
            if object_id not in indices:
                print(anno_id)


if __name__ == '__main__':
    # create_correspondences()
    verify_correspondence()
    # copy_correspondence()
