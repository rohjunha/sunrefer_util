import json
from multiprocessing import Pool
from pathlib import Path
from typing import List

import cv2
import numpy as np
import open3d as o3d
import torch
from matplotlib import pyplot as plt
from pytorch3d.ops import box3d_overlap
from tqdm import tqdm

from utils.intrinsic_fetcher import IntrinsicFetcher
from utils.line_mesh import LineMesh
from utils.meta_io import fetch_scene_object_by_image_id
from utils.pred_and_anno import fetch_predicted_bbox_by_image_id


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
    line_mesh = LineMesh(pl, lines=lines, colors=color, radius=radius)
    line_mesh.translate(centroid)
    return line_mesh


def rearrange_points(pts):
    return np.array([pts[i] for i in [0, 1, 3, 2, 4, 5, 7, 6]])


def line_mesh_iou(a: LineMesh, b: LineMesh):
    aa = rearrange_points(a.points)
    bb = rearrange_points(b.points)
    return box3d_overlap(
        torch.from_numpy(aa[np.newaxis, ...].astype(np.float32)),
        torch.from_numpy(bb[np.newaxis, ...].astype(np.float32)))


def select_inlier_pcd(pcd, pcd_th_ratio: float):
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
    num_points = labels.shape[0]
    num_count_threshold = int(pcd_th_ratio * num_points)
    inlier_labels = []
    for l in range(np.min(labels), np.max(labels) + 1):
        count = np.sum(labels == l)
        if count > num_count_threshold:
            inlier_labels.append(l)
    indices = np.where(np.isin(labels, inlier_labels))[0]
    return pcd.select_by_index(indices)


def compute_iou(
        uniq_id, scene,
        fx, fy, cx, cy, height, width,
        color_raw, depth_raw, seg_out_mask,
        bbox_2d: List[float],
        aabb: bool,
        bbox_2d_ratio: float = 0.8,
        pcd_th_ratio: float = 0.1,
        mask_ratio: float = 0.8):
    try:
        image_id, object_id, anno_id = uniq_id.split('_')
        object_id = int(object_id)

        x1, y1, w, h = bbox_2d
        x1 = int(x1 + 0.5 * w * (1 - bbox_2d_ratio))
        x2 = int(x1 + 0.5 * w * (1 + bbox_2d_ratio))
        y1 = int(y1 + 0.5 * h * (1 - bbox_2d_ratio))
        y2 = int(y1 + 0.5 * h * (1 + bbox_2d_ratio))

        color_np = np.asarray(color_raw, dtype=np.uint8)
        depth_np = np.asarray(depth_raw, dtype=np.uint16)
        new_color_np = np.zeros_like(color_np)
        new_depth_np = np.zeros_like(depth_np)
        new_color_np[y1:y2, x1:x2, :] = color_np[y1:y2, x1:x2, :]
        new_depth_np[y1:y2, x1:x2] = depth_np[y1:y2, x1:x2]
        if seg_out_mask is not None:
            if np.sum(seg_out_mask[y1:y2, x1:x2].astype(np.int64)) < (y2 - y1) * (x2 - x1) * mask_ratio:
                new_color_np[seg_out_mask, :] = 0
                new_depth_np[seg_out_mask] = 0.

        new_color = o3d.geometry.Image(np.ascontiguousarray(new_color_np).astype(np.uint8))
        new_depth = o3d.geometry.Image(np.ascontiguousarray(new_depth_np).astype(np.uint16))
        cropped_rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(new_color, new_depth,
                                                                           convert_rgb_to_intensity=False)

        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = scene.extrinsics

        Tyz = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        iTyz = np.linalg.inv(Tyz)

        # Computes a pointcloud from the cropped rgbd image.
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            cropped_rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy))
        pcd.transform(Rt)

        # Filter out outliers in small clusters.
        inlier_pcd = select_inlier_pcd(pcd, pcd_th_ratio) if pcd_th_ratio > 1e-3 else pcd

        # Computes a bounding box from the inlier pointcloud.
        # Converts the bbox to canonical LineMesh form.
        bbox_from_pcd = inlier_pcd.get_axis_aligned_bounding_box()
        # bbox_from_pcd = inlier_pcd.get_oriented_bounding_box()
        bbox_from_pcd.color = (0, 1, 1)
        bbox_points = np.asarray(bbox_from_pcd.get_box_points())
        bbox_points = np.array([bbox_points[i] for i in [0, 1, 2, 7, 3, 6, 5, 4]])
        bbox_mesh = LineMesh(bbox_points, colors=(0, 1, 1), radius=0.03)

        # Computes the LineMesh from the ground-truth target bbox.
        bbox_3d = scene.gt_3d_bbox[int(object_id)]
        centroid = bbox_3d.centroid[0]  # (3, )
        basis = bbox_3d.basis  # (3, 3)
        coeffs = bbox_3d.coeffs[0]  # (3, )
        target_mesh = create_line_mesh(centroid, basis, coeffs, aabb=aabb)
        target_mesh.transform(iTyz)

        _, iou = line_mesh_iou(bbox_mesh, target_mesh)
        return iou.item()
    except:
        print('found an error: {}'.format(uniq_id))
        return 0.


def func(x):
    return x[0], compute_iou(*x)


def compute_average_iou(
        aabb: bool,
        use_seg_out_mask: bool,
        bbox_2d_ratio: float,
        pcd_th_ratio: float,
        mask_ratio: float):
    scene_by_image_id = fetch_scene_object_by_image_id('v2_3d')
    intrinsic_fetcher = IntrinsicFetcher()
    pred_bbox_by_image_id = fetch_predicted_bbox_by_image_id()
    seg_out_dir = Path('/home/junha/Downloads/segformer')

    # Prepare items.
    items = []
    for image_id, pred_items in tqdm(list(pred_bbox_by_image_id.items())):
        scene = scene_by_image_id[image_id]
        fx, fy, cx, cy, height, width = intrinsic_fetcher[image_id]
        color_raw = cv2.imread(str(scene.rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(scene.depth_path), cv2.IMREAD_UNCHANGED)
        if use_seg_out_mask:
            seg_out = np.load(str(seg_out_dir / '{}.npy'.format(image_id)))
            seg_out_mask = np.zeros(seg_out.shape, dtype=bool)
            for label in [0, 3, 5]:
                seg_out_mask[seg_out == label] = True
        else:
            seg_out_mask = None
        for uniq_id, bbox in pred_items:
            items.append((uniq_id, scene, fx, fy, cx, cy, height, width,
                          color_raw, depth_raw, seg_out_mask, bbox,
                          aabb, bbox_2d_ratio, pcd_th_ratio, mask_ratio))

    pool = Pool(12)
    result_list_tqdm = []
    for result in tqdm(pool.imap_unordered(func, items), total=len(items)):
        result_list_tqdm.append(result)
    iou_by_uniq_id = {k: v for k, v in result_list_tqdm}

    filename = 'aabb={},seg_out={},bbox_ratio={},pcd_th={},mask_ratio={}.json'.format(
        aabb, use_seg_out_mask, bbox_2d_ratio, pcd_th_ratio, mask_ratio)
    with open('/home/junha/data/sunrefer/iou/{}'.format(filename), 'w') as file:
        json.dump(iou_by_uniq_id, file, indent=4)

    iou_25 = 0
    iou_50 = 0
    for iou in iou_by_uniq_id.values():
        if iou > 0.25:
            iou_25 += 1
        if iou > 0.5:
            iou_50 += 1
    iou_25 = iou_25 / len(iou_by_uniq_id) * 100
    iou_50 = iou_50 / len(iou_by_uniq_id) * 100
    print('acc@0.25: {:5.3f}'.format(iou_25))
    print('acc@0.50: {:5.3f}'.format(iou_50))


class PredictionVisualizer:
    def __init__(self,
                 aabb: bool,
                 highlight: bool,
                 apply_seg_out_mask: bool,
                 verbose: bool = False,
                 bbox_2d_ratio: float = 0.8,
                 pcd_th_ratio: float = 0.1,
                 mask_ratio: float = 0.8):
        self.aabb = aabb
        self.highlight = highlight
        self.apply_seg_out_mask = apply_seg_out_mask
        self.verbose = verbose
        self.bbox_2d_ratio = bbox_2d_ratio
        self.pcd_th_ratio = pcd_th_ratio
        self.mask_ratio = mask_ratio

        self.scene_by_image_id = fetch_scene_object_by_image_id('v2_3d')
        self.intrinsic_fetcher = IntrinsicFetcher()
        self.pred_bbox_by_image_id = fetch_predicted_bbox_by_image_id()
        self.seg_out_dir = Path('/home/junha/Downloads/segformer')
        self.exception_label_indices = [0, 3, 5]

    def fetch_seg_out_mask(self, image_id: str):
        seg_out = np.load(str(self.seg_out_dir / '{}.npy'.format(image_id)))
        seg_out_mask = np.zeros(seg_out.shape, dtype=bool)
        for label in self.exception_label_indices:
            seg_out_mask[seg_out == label] = True
        return seg_out_mask

    def compute_3d_bbox_by_image_id(self, image_id: str, anno_index: int):
        uniq_id, bbox_2d = self.pred_bbox_by_image_id[image_id][anno_index]
        return self.compute_3d_bbox(uniq_id, bbox_2d)

    def compute_3d_bbox(self, uniq_id: str, bbox_2d: List[float]):
        try:
            image_id, object_id, anno_id = uniq_id.split('_')
            object_id = int(object_id)
            scene = self.scene_by_image_id[image_id]
            fx, fy, cx, cy, height, width = self.intrinsic_fetcher[image_id]

            color_raw = cv2.imread(str(scene.rgb_path), cv2.IMREAD_COLOR)
            depth_raw = cv2.imread(str(scene.depth_path), cv2.IMREAD_UNCHANGED)
            # rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(
            #     o3d.geometry.Image(color_raw),
            #     o3d.geometry.Image(depth_raw),
            #     convert_rgb_to_intensity=False)

            color_np = np.asarray(color_raw, dtype=np.uint8)
            depth_np = np.asarray(depth_raw, dtype=np.uint16)

            x1, y1, w, h = bbox_2d
            x1 = int(x1 + 0.5 * w * (1 - self.bbox_2d_ratio))
            x2 = int(x1 + 0.5 * w * (1 + self.bbox_2d_ratio))
            y1 = int(y1 + 0.5 * h * (1 - self.bbox_2d_ratio))
            y2 = int(y1 + 0.5 * h * (1 + self.bbox_2d_ratio))

            new_color_np = np.zeros_like(color_np)
            new_depth_np = np.zeros_like(depth_np)
            new_color_np[y1:y2, x1:x2, :] = color_np[y1:y2, x1:x2, :]
            new_depth_np[y1:y2, x1:x2] = depth_np[y1:y2, x1:x2]

            if self.apply_seg_out_mask:
                seg_out_mask = self.fetch_seg_out_mask(image_id)
                if np.sum(seg_out_mask[y1:y2, x1:x2].astype(np.int64)) < (y2 - y1) * (x2 - x1) * self.mask_ratio:
                    new_color_np[seg_out_mask, :] = 0
                    new_depth_np[seg_out_mask] = 0.

            new_color = o3d.geometry.Image(np.ascontiguousarray(new_color_np).astype(np.uint8))
            new_depth = o3d.geometry.Image(np.ascontiguousarray(new_depth_np).astype(np.uint16))
            new_rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(new_color, new_depth,
                                                                           convert_rgb_to_intensity=False)

            # import matplotlib.pyplot as plt
            # plt.subplot(1, 2, 1)
            # plt.title('SUN grayscale image')
            # plt.imshow(new_rgbd_image.color)
            # plt.subplot(1, 2, 2)
            # plt.title('SUN depth image')
            # plt.imshow(new_rgbd_image.depth)
            # plt.show()

            Rt = np.eye(4, dtype=np.float32)
            Rt[:3, :3] = scene.extrinsics

            Tyz = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
            iTyz = np.linalg.inv(Tyz)
            tr_flip = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                new_rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy))
            pcd.transform(Rt)
            pcd.transform(tr_flip)

            if self.pcd_th_ratio > 1e-3:
                labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))

                max_label = labels.max()
                colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
                colors[labels < 0] = 0
                pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

                num_points = labels.shape[0]
                num_count_threshold = int(self.pcd_th_ratio * num_points)
                inlier_labels = []
                for l in range(np.min(labels), np.max(labels) + 1):
                    count = np.sum(labels == l)
                    if count > num_count_threshold:
                        inlier_labels.append(l)
                indices = np.where(np.isin(labels, inlier_labels))[0]
                inlier_pcd = pcd.select_by_index(indices)
            else:
                inlier_pcd = pcd

            aabb_from_pcd = inlier_pcd.get_axis_aligned_bounding_box()
            aabb_from_pcd.color = (0, 1, 1)
            aabb_points = np.asarray(aabb_from_pcd.get_box_points())
            aabb_points = np.array([aabb_points[i] for i in [0, 1, 2, 7, 3, 6, 5, 4]])
            bbox_mesh = LineMesh(aabb_points, colors=(0, 1, 1), radius=0.03)

            if self.verbose:
                o3d_obj_list = []
                o3d_obj_list += bbox_mesh.cylinder_segments
                o3d_obj_list.append(pcd)

                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
                o3d_obj_list.append(coord_frame)

                target_mesh = None
                for oid, bbox_3d in enumerate(scene.gt_3d_bbox):
                    # class_name = bbox_3d.class_name[0]  # str
                    centroid = bbox_3d.centroid[0]  # (3, )
                    basis = bbox_3d.basis  # (3, 3)
                    coeffs = bbox_3d.coeffs[0]  # (3, )
                    # orientation = bbox_3d.orientation[0]  # (3, )

                    if self.highlight and oid == object_id:
                        color = [0, 1, 0]
                        radius = 0.03
                    else:
                        color = [1, 0, 0]
                        radius = 0.015

                    line_mesh = create_line_mesh(centroid, basis, coeffs, aabb=self.aabb, color=color, radius=radius)
                    line_mesh.transform(iTyz)
                    line_mesh.transform(tr_flip)
                    o3d_obj_list += line_mesh.cylinder_segments
                    if oid == object_id:
                        target_mesh = line_mesh
                o3d.visualization.draw_geometries(o3d_obj_list)
            else:
                bbox_3d = scene.gt_3d_bbox[object_id]
                centroid = bbox_3d.centroid[0]  # (3, )
                basis = bbox_3d.basis  # (3, 3)
                coeffs = bbox_3d.coeffs[0]  # (3, )
                target_mesh = create_line_mesh(centroid, basis, coeffs, aabb=self.aabb)
                target_mesh.transform(iTyz)
                target_mesh.transform(tr_flip)

            print(target_mesh.points)
            print(bbox_mesh.points)

            overlap_volume, iou = line_mesh_iou(bbox_mesh, target_mesh)
            return iou.item()
        except:
            print('found an error during computing iou: {}'.format(uniq_id, bbox_2d))
            return 0.


if __name__ == '__main__':
    vis = PredictionVisualizer(
        aabb=True,
        highlight=True,
        apply_seg_out_mask=True,
        verbose=True,
        bbox_2d_ratio=1.0,
        pcd_th_ratio=0.1,
        mask_ratio=0.8)
    print(vis.compute_3d_bbox_by_image_id('000001', 0))
    # compute_average_iou(aabb=True,
    #                     use_seg_out_mask=True,
    #                     bbox_2d_ratio=1.0,
    #                     pcd_th_ratio=0.1,
    #                     mask_ratio=0.8)