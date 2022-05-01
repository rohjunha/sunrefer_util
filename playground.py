import json
from typing import List

import cv2
import numpy as np

from utils.directory import fetch_depth_data_path, fetch_ofa_predict_path
from utils.image_io import fetch_depth_data, fetch_rgb, PointcloudConverter
from utils.intrinsic_fetcher import IntrinsicFetcher
from utils.line_mesh import LineMesh
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


def convert_orientedbbox2AABB(verts):
    # c_x = all_bboxes[:, 0]
    # c_y = all_bboxes[:, 1]
    # c_z = all_bboxes[:, 2]
    # s_x = all_bboxes[:, 3]
    # s_y = all_bboxes[:, 4]
    # s_z = all_bboxes[:, 5]
    # angle = all_bboxes[:, 6]
    # orientation = np.concatenate([np.cos(angle)[:, np.newaxis],
    #                               -np.sin(angle)[:, np.newaxis]], axis=1)
    # ori1 = orientation
    # ori2 = np.ones(ori1.shape)
    # ori2 = ori2 - np.sum(ori1 * ori2, axis=1)[:, np.newaxis] * ori1
    # ori2 = ori2 / np.linalg.norm(ori2, axis=1)[:, np.newaxis]
    # ori1 = ori1 * s_x[:, np.newaxis]
    # ori2 = ori2 * s_y[:, np.newaxis]
    # verts = np.array([[c_x, c_y, c_z - s_z / 2],
    #                   [c_x, c_y, c_z + s_z / 2],
    #                   [c_x, c_y, c_z - s_z / 2],
    #                   [c_x, c_y, c_z + s_z / 2],
    #                   [c_x, c_y, c_z - s_z / 2],
    #                   [c_x, c_y, c_z + s_z / 2],
    #                   [c_x, c_y, c_z - s_z / 2],
    #                   [c_x, c_y, c_z + s_z / 2]])
    # verts = verts.transpose(2, 0, 1)
    # verts[:, 0, 0:2] = verts[:, 0, 0:2] - ori2 / 2 - ori1 / 2
    # verts[:, 1, 0:2] = verts[:, 1, 0:2] - ori2 / 2 - ori1 / 2
    # verts[:, 2, 0:2] = verts[:, 2, 0:2] - ori2 / 2 + ori1 / 2
    # verts[:, 3, 0:2] = verts[:, 3, 0:2] - ori2 / 2 + ori1 / 2
    # verts[:, 4, 0:2] = verts[:, 4, 0:2] + ori2 / 2 - ori1 / 2
    # verts[:, 5, 0:2] = verts[:, 5, 0:2] + ori2 / 2 - ori1 / 2
    # verts[:, 6, 0:2] = verts[:, 6, 0:2] + ori2 / 2 + ori1 / 2
    # verts[:, 7, 0:2] = verts[:, 7, 0:2] + ori2 / 2 + ori1 / 2
    x_min = np.min(verts[:, 0], axis=0)
    x_max = np.max(verts[:, 0], axis=0)
    y_min = np.min(verts[:, 1], axis=0)
    y_max = np.max(verts[:, 1], axis=0)
    z_min = np.min(verts[:, 2], axis=0)
    z_max = np.max(verts[:, 2], axis=0)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    cz = (z_min + z_max) / 2
    sx = x_max - x_min
    sy = y_max - y_min
    sz = z_max - z_min
    return [cx, cy, cz, sx, sy, sz]


def create_line_mesh(centroid, basis, coeffs, aabb: bool, color=[1, 0, 0], radius=0.03):
    dx = basis[0] * coeffs[0]
    dy = basis[1] * coeffs[1]
    dz = basis[2] * coeffs[2]

    cl = [[-1, -1, -1],
          [1, -1, -1],
          [-1, 1, -1],
          [1, 1, -1],
          [-1, -1, 1],
          [1, -1, 1],
          [-1, 1, 1],
          [1, 1, 1]]
    pl = [sx * dx + sy * dy + sz * dz for sx, sy, sz in cl]

    if aabb:
        cx, cy, cz, sx, sy, sz = convert_orientedbbox2AABB(np.array(pl))
        pl = [[cx + 0.5 * dx * sx, cy + 0.5 * dy * sy, cz + 0.5 * dz * sz] for dx, dy, dz in cl]

    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    line_mesh2 = LineMesh(pl, lines=lines, colors=color, radius=radius)
    line_mesh2.translate(centroid)
    return line_mesh2


def draw_3d_bbox(uniq_id: str, bbox_2d: List[float], aabb: bool, highlight: bool):
    BBOX_2D_RATIO = 0.8
    PCD_TH_RATIO = 0.1

    image_id, object_id, anno_id = uniq_id.split('_')
    object_id = int(object_id)
    scene = SCENE_BY_IMAGE_ID[image_id]
    fx, fy, cx, cy, height, width = INTRINSIC_FETCHER[image_id]

    color_raw = cv2.imread(str(scene.rgb_path), cv2.IMREAD_COLOR)
    depth_raw = cv2.imread(str(scene.depth_path), cv2.IMREAD_UNCHANGED)
    rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(o3d.geometry.Image(color_raw), o3d.geometry.Image(depth_raw), convert_rgb_to_intensity=False)
    print(color_raw.shape, color_raw.dtype)
    print(depth_raw.shape, depth_raw.dtype)
    print(np.min(depth_raw))
    print(np.max(depth_raw))

    # color_raw = o3d.io.read_image(str(scene.rgb_path))
    # depth_raw = o3d.io.read_image(str(scene.depth_path))
    # rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(color_raw, depth_raw, convert_rgb_to_intensity=False)
    print(type(color_raw))
    print(type(rgbd_image.color))
    print(np.min(depth_raw))
    print(np.max(depth_raw))
    print(np.min(rgbd_image.depth))
    print(np.max(rgbd_image.depth))

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.title('SUN grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('SUN depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

    # todo: convert raw images into np.ndarray (from np.asarray)
    # and crop the image w.r.t the predicted bbox_2d,
    # convert them back to pcd,
    # create a minimum 3d bounding box.

    color_np = np.asarray(color_raw, dtype=np.uint8)
    depth_np = np.asarray(depth_raw, dtype=np.uint16)
    print(color_np.shape, color_np.dtype)
    print(depth_np.shape, depth_np.dtype)
    print(np.min(depth_np))
    print(np.max(depth_np))


    x1, y1, w, h = list(map(int, bbox_2d))
    x2, y2 = x1 + w, y1 + h


    x1, y1, w, h = bbox_2d
    x1 = int(x1 + 0.5 * w * (1 - BBOX_2D_RATIO))
    x2 = int(x1 + 0.5 * w * (1 + BBOX_2D_RATIO))
    y1 = int(y1 + 0.5 * h * (1 - BBOX_2D_RATIO))
    y2 = int(y1 + 0.5 * h * (1 + BBOX_2D_RATIO))

    new_color_np = np.zeros_like(color_np)
    new_depth_np = np.zeros_like(depth_np)
    new_color_np[y1:y2, x1:x2, :] = color_np[y1:y2, x1:x2, :]
    new_depth_np[y1:y2, x1:x2] = depth_np[y1:y2, x1:x2]
    print(np.min(new_depth_np))
    print(np.max(new_depth_np))
    new_color = o3d.geometry.Image(np.ascontiguousarray(new_color_np).astype(np.uint8))
    new_depth = o3d.geometry.Image(np.ascontiguousarray(new_depth_np).astype(np.uint16))
    new_rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(new_color, new_depth, convert_rgb_to_intensity=False)

    print(np.min(new_depth))
    print(np.max(new_depth))
    print(np.min(new_rgbd_image.depth))
    print(np.max(new_rgbd_image.depth))

    # o3d.geometry.AxisAlignedBoundingBox()
    #
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.title('SUN grayscale image')
    plt.imshow(new_rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('SUN depth image')
    plt.imshow(new_rgbd_image.depth)
    plt.show()

    # return

    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = scene.extrinsics
    iRt = np.linalg.inv(Rt)

    Tyz = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    iTyz = np.linalg.inv(Tyz)
    tr_flip = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

    o3d_obj_list = []
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        new_rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy))

    # plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
    #                                          ransac_n=3,
    #                                          num_iterations=1000)
    # [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    # inlier_cloud = pcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # o3d_obj_list.append(inlier_cloud)
    # o3d_obj_list.append(outlier_cloud)

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    num_points = labels.shape[0]
    num_count_threshold = int(PCD_TH_RATIO * num_points)
    inlier_labels = []

    print(labels.shape, np.min(labels), np.max(labels))
    for l in range(np.min(labels), np.max(labels) + 1):
        count = np.sum(labels == l)
        if count > num_count_threshold:
            inlier_labels.append(l)
        print(l, np.sum(labels == l))
    indices = np.where(np.isin(labels, inlier_labels))[0]

    inlier_pcd = pcd.select_by_index(indices)
    aabb_from_pcd = inlier_pcd.get_axis_aligned_bounding_box()
    aabb_from_pcd.color = (0, 1, 1)
    aabb_points = np.asarray(aabb_from_pcd.get_box_points())
    print(aabb_points)

    # cl = [[-1, -1, -1], >> 0
    #       [1, -1, -1], >> 1
    #       [-1, 1, -1],  2
    #       [1, 1, -1],   3
    #       [-1, -1, 1],  4
    #       [1, -1, 1],   5
    #       [-1, 1, 1],   6
    #       [1, 1, 1]]    7
    #
    # nl = [[-1, -1, -1], 0 0
    #       [1, -1, -1],  1 1
    #       [-1, 1, -1],  2 2
    #       [-1, -1, 1],  4 3
    #       [1, 1, 1],    7 4
    #       [-1, 1, 1],   6 5
    #       [1, -1, 1],   5 6
    #       [1, 1, -1]]   3 7
    # canonical index: 0, 1, ..., 7
    # new point index: 0, 1, 2, 4,

    aabb_points = np.array([aabb_points[i] for i in [0, 1, 2, 7, 3, 6, 5, 4]])

    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    bbox_mesh = LineMesh(aabb_points, lines=lines, colors=(0, 1, 1), radius=0.03)
    bbox_mesh.transform(tr_flip)
    o3d_obj_list += bbox_mesh.cylinder_segments

    # inlier_pcd.transform(tr_flip)
    # # o3d_obj_list.append(aabb_from_pcd)
    # o3d_obj_list.append(inlier_pcd)

    pcd.transform(tr_flip)
    o3d_obj_list.append(pcd)

    # Flip it, otherwise the pointcloud will be upside down
    # pcd.transform(Rt)
    # pcd.transform(tr_yz)




    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    # coord_frame.transform(tr_yz)
    o3d_obj_list.append(coord_frame)

    for oid, bbox_3d in enumerate(scene.gt_3d_bbox):
        class_name = bbox_3d.class_name[0]  # str
        centroid = bbox_3d.centroid[0]  # (3, )
        basis = bbox_3d.basis  # (3, 3)
        coeffs = bbox_3d.coeffs[0]  # (3, )
        orientation = bbox_3d.orientation[0]  # (3, )

        if highlight and oid == object_id:
            color = [0, 1, 0]
            radius = 0.03
        else:
            color = [1, 0, 0]
            radius = 0.015

        line_mesh = create_line_mesh(centroid, basis, coeffs, aabb=aabb, color=color, radius=radius)
        line_mesh.transform(iTyz)
        line_mesh.transform(iRt)
        line_mesh.transform(tr_flip)
        o3d_obj_list += line_mesh.cylinder_segments

    o3d.visualization.draw_geometries(o3d_obj_list)


def test_box():
    from pytorch3d.ops import box3d_overlap
    UNIT_BOX = [
        [0, 0, 0],  # 0
        [1, 0, 0],  # 1
        [1, 1, 0],  # 3 > 2
        [0, 1, 0],  # 2 > 3
        [0, 0, 1],  # 4
        [1, 0, 1],  # 5
        [1, 1, 1],  # 7 > 6
        [0, 1, 1],  # 6 > 7
    ]

    cl = [[-1, -1, -1],  # 0
          [1, -1, -1],   # 1
          [-1, 1, -1],   # 2
          [1, 1, -1],    # 3
          [-1, -1, 1],   # 4
          [1, -1, 1],    # 5
          [-1, 1, 1],    # 6
          [1, 1, 1]]     # 7

    # nl = [[-1, -1, -1], 0
    #       [1, -1, -1],  1
    #       [-1, 1, -1],  3
    #       [-1, -1, 1],  4
    #       [1, 1, 1],    6
    #       [-1, 1, 1],   7
    #       [1, -1, 1],   5
    #       [1, 1, -1]]   2

    w, h, d = 0.5, 0.5, 0.5
    b1 = np.array(UNIT_BOX)

    w, h, d = 0.5, 0.5, 0.5
    b2 = np.array(UNIT_BOX) * 0.9
    print(b1.shape, b2.shape)
    print(b1)
    print(b2)

    o3d_obj_list = []
    # lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    lines = [[0, 1], [0, 3], [1, 2], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7], [0, 4], [1, 5], [3, 7], [2, 6]]
    l1 = LineMesh(b1, lines=lines, colors=(1, 0, 0), radius=0.03)
    l2 = LineMesh(b2, lines=lines, colors=(0, 1, 0), radius=0.03)
    o3d_obj_list += l1.cylinder_segments
    o3d_obj_list += l2.cylinder_segments
    o3d.visualization.draw_geometries(o3d_obj_list)

    import torch
    intersection_vol, iou_3d = box3d_overlap(
        torch.from_numpy(b1[np.newaxis, ...].astype(np.float32)),
        torch.from_numpy(b2[np.newaxis, ...].astype(np.float32)))
    print(intersection_vol, iou_3d)


if __name__ == '__main__':
    test_box()

    # scene_by_image_id = fetch_scene_object_by_image_id('v1')
    # for i in range(1, 10):
    #     print(i, scene_by_image_id['{:06d}'.format(i)].gt_corner_3d.shape)


    # image_id = '000001'
    # '002920'
    # pred_bbox_by_uniq_id = fetch_predicted_bbox_by_uniq_id()
    # for uniq_id, bbox in list(pred_bbox_by_uniq_id.items()):
    #     if uniq_id.startswith('005001'):
    #         print(uniq_id)
    #         draw_3d_bbox(uniq_id, bbox, aabb=True, highlight=True)
    #         break
