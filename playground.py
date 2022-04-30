import json
from typing import List

import cv2
import numpy as np

from utils.directory import fetch_depth_data_path, fetch_ofa_predict_path
from utils.image_io import fetch_depth_data, fetch_rgb, PointcloudConverter
from utils.intrinsic_fetcher import IntrinsicFetcher
from utils.meta_io import fetch_scene_object_by_image_id
import open3d as o3d



SCENE_BY_IMAGE_ID = fetch_scene_object_by_image_id('v2_3d')
INTRINSIC_FETCHER = IntrinsicFetcher()


def fetch_predicted_bbox_by_uniq_id():
    with open(str(fetch_ofa_predict_path()), 'r') as file:
        pred_data = json.load(file)
    pred_bbox_by_uniq_id = dict()
    for pred_item in pred_data:
        bbox = pred_item['box']
        pred_bbox_by_uniq_id[pred_item['uniq_id']] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    return pred_bbox_by_uniq_id


def create_line_set(centroid, basis, coeffs, orientation):
    print("Let's draw a cubic using o3d.geometry.LineSet.")

    dx = basis[0] * coeffs[0]
    dy = basis[1] * coeffs[1]
    dz = basis[2] * coeffs[2]
    theta = np.math.atan2(orientation[1], orientation[0])
    c, s = np.math.cos(theta), np.math.sin(theta)

    cl = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    pl = [(2 * (sx - 0.5) * dx + 2 * (sy - 0.5) * dy + 2 * (sz - 0.5) * dz) for sx, sy, sz in cl]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pl),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    line_set.transform([[1, 0, 0, centroid[0]],
                        [0, 1, 0, centroid[1]],
                        [0, 0, 1, centroid[2]],
                        [0, 0, 0, 1]])
    # line_set.transform([[c, -s, 0, centroid[0]],
    #                     [s, c, 0, centroid[1]],
    #                     [0, 0, 1, centroid[2]],
    #                     [0, 0, 0, 1]])
    tr_yz = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    # line_set.transform(tr_yz)
    return line_set


def draw_3d_bbox(uniq_id: str, bbox_2d: List[float]):
    image_id, object_id, anno_id = uniq_id.split('_')
    scene = SCENE_BY_IMAGE_ID[image_id]
    fx, fy, cx, cy, height, width = INTRINSIC_FETCHER[image_id]
    color_raw = o3d.io.read_image(str(scene.rgb_path))
    depth_raw = o3d.io.read_image(str(scene.depth_path))
    rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(color_raw, depth_raw, convert_rgb_to_intensity=False)

    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = scene.extrinsics

    tr_yz = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    tr_flip = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

    o3d_obj_list = []
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform(Rt)
    pcd.transform(tr_yz)
    o3d_obj_list.append(pcd)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    # coord_frame.transform(tr_yz)
    o3d_obj_list.append(coord_frame)

    for bbox_3d in scene.gt_3d_bbox:
        class_name = bbox_3d.class_name[0]  # str
        centroid = bbox_3d.centroid[0]  # (3, )
        basis = bbox_3d.basis  # (3, 3)
        coeffs = bbox_3d.coeffs[0]  # (3, )
        orientation = bbox_3d.orientation[0]  # (3, )

        print('class_name', class_name)
        print('centroid', centroid)
        print('basis', basis)
        print('coeffs', coeffs)
        print('orientation', orientation)
        print()


        # mesh_box = o3d.geometry.TriangleMesh.create_box(width=coeffs[0], height=coeffs[1], depth=coeffs[2])
        # o3d_obj_list.append(mesh_box)
        o3d_obj_list.append(create_line_set(centroid, basis, coeffs, orientation))
    o3d.visualization.draw_geometries(o3d_obj_list)

    # pccvt = PointcloudConverter()
    # pc, mask = pccvt[image_id]
    # pc = o3d.geometry.PointCloud(points=pc.reshape((-1, 3)))
    # o3d.io.write_point_cloud('/home/junha/Downloads/test.obj', pc)
    # o3d.visualization.draw_geometries([pc])
    # print(pc.shape, mask.shape)
    # cv2.imshow('pc', pc)
    # cv2.waitKey()
    # print(image_id)
    # im = IntrinsicFetcher()
    # scene_3d = fetch_scene_object_by_image_id('v2_3d')[image_id]
    # print(im[image_id])
    # print(scene_3d)
    # fetch_depth_data_path(image_id)


if __name__ == '__main__':
    # scene_by_image_id = fetch_scene_object_by_image_id('v1')
    # for i in range(1, 10):
    #     print(i, scene_by_image_id['{:06d}'.format(i)].gt_corner_3d.shape)


    # image_id = '000001'
    pred_bbox_by_uniq_id = fetch_predicted_bbox_by_uniq_id()
    for uniq_id, bbox in list(pred_bbox_by_uniq_id.items())[20:21]:
        draw_3d_bbox(uniq_id, bbox)
