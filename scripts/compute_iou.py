import json
from multiprocessing import Pool
from pathlib import Path
from typing import List, Union, Tuple, Dict, Any

import cv2
import numpy as np
import open3d as o3d
import torch
from matplotlib import pyplot as plt
from pytorch3d.ops import box3d_overlap
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.directory import fetch_segformer_path, fetch_xyzrgb_pcd_path, fetch_xyzrgb_bbox_path, fetch_split_test_path, \
    fetch_split_train_path, fetch_object_2d_3d_path, fetch_sunrefer_anno_path
from utils.intrinsic_fetcher import IntrinsicFetcher
from utils.line_mesh import LineMesh
from utils.meta_io import fetch_scene_object_by_image_id, MetaObject2DV2, MetaObject3DV2
from utils.pred_and_anno import fetch_predicted_bbox_by_image_id


def convert_raw_depth(raw_depth):
    """Numpy implementation of SUNRGBD depth processing."""
    shift_depth = np.bitwise_or(np.right_shift(raw_depth, 3), np.left_shift(raw_depth, 13))
    float_depth = shift_depth.astype(dtype=np.float32) / 1000.
    float_depth[float_depth > 7.0] = 0.
    return float_depth


def convert_pcd(raw_depth, extrinsics, fx, fy, cx, cy):
    m = raw_depth > 0
    zz = convert_raw_depth(raw_depth)
    mx = np.linspace(0, raw_depth.shape[1] - 1, raw_depth.shape[1])
    my = np.linspace(0, raw_depth.shape[0] - 1, raw_depth.shape[0])
    rx = np.linspace(0., 1., raw_depth.shape[1])
    ry = np.linspace(0., 1., raw_depth.shape[0])
    rx, ry = np.meshgrid(rx, ry)
    xx, yy = np.meshgrid(mx, my)
    xx = (xx - cx) * zz / fx
    yy = (yy - cy) * zz / fy
    pcd = np.stack((xx, yy, zz), axis=-1) @ np.transpose(extrinsics)
    return pcd, m, rx, ry


def aabb_from_oriented_bbox(verts):
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


LINES = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]


def create_line_mesh(
        centroid: np.array,
        basis: np.array,
        coeffs: np.array,
        aabb: bool,
        color: Tuple[float, float, float] = (1., 0., 0.),
        radius: float = 0.015):
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
        cx, cy, cz, sx, sy, sz = aabb_from_oriented_bbox(np.array(pl))
        pl = [[cx + 0.5 * dx * sx, cy + 0.5 * dy * sy, cz + 0.5 * dz * sz] for dx, dy, dz in cl]

    line_mesh = LineMesh(pl, lines=LINES, colors=color, radius=radius)
    line_mesh.translate(centroid)
    return line_mesh


def create_line_mesh_from_dict(
        bbox_dict: Dict[str, Any],
        aabb: bool,
        color: Tuple[float, float, float] = (1., 0., 0.),
        radius: float = 0.015) -> LineMesh:
    centroid = np.reshape(bbox_dict['centroid'], (3, ))
    basis = bbox_dict['basis']
    coeffs = np.reshape(bbox_dict['coeffs'], (3, ))
    return create_line_mesh(
        centroid=centroid,
        basis=basis,
        coeffs=coeffs,
        aabb=aabb,
        color=color,
        radius=radius)


def create_line_mesh_from_center_size(
        center_size: np.ndarray,
        color: Union[List[int], Tuple[int, int, int]] = (1, 0, 0),
        radius: float = 0.015):
    center, hsz = center_size[:3], center_size[3:6] * 0.5
    points = [[-hsz[0], -hsz[1], -hsz[2]],
              [+hsz[0], -hsz[1], -hsz[2]],
              [-hsz[0], +hsz[1], -hsz[2]],
              [+hsz[0], +hsz[1], -hsz[2]],
              [-hsz[0], -hsz[1], +hsz[2]],
              [+hsz[0], -hsz[1], +hsz[2]],
              [-hsz[0], +hsz[1], +hsz[2]],
              [+hsz[0], +hsz[1], +hsz[2]]]
    line_mesh = LineMesh(points, lines=LINES, colors=color, radius=radius)
    line_mesh.translate(center)
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

    # Prepare items.
    items = []
    for image_id, pred_items in tqdm(list(pred_bbox_by_image_id.items())):
        scene = scene_by_image_id[image_id]
        fx, fy, cx, cy, height, width = intrinsic_fetcher[image_id]
        color_raw = cv2.imread(str(scene.rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(scene.depth_path), cv2.IMREAD_UNCHANGED)
        if use_seg_out_mask:
            seg_out = np.load(str(fetch_segformer_path(image_id)))
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


def normalize_rgb(raw_rgb):
    float_rgb = np.array(raw_rgb).astype(np.float32) / 255.
    float_rgb[..., 0] = (float_rgb[..., 0] - 0.485) / 0.229
    float_rgb[..., 1] = (float_rgb[..., 1] - 0.456) / 0.224
    float_rgb[..., 2] = (float_rgb[..., 2] - 0.406) / 0.225
    return float_rgb


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
        self.cls_aabb_list = json.load(open(str(fetch_xyzrgb_bbox_path()), 'r'))

    def fetch_seg_out_mask(self, image_id: str):
        seg_out = np.load(str(fetch_segformer_path(image_id)))
        seg_out_mask = np.zeros(seg_out.shape, dtype=bool)
        for label in self.exception_label_indices:
            seg_out_mask[seg_out == label] = True
        return seg_out_mask

    def compute_3d_bbox_by_image_id(self, image_id: str, anno_index: int):
        uniq_id, bbox_2d = self.pred_bbox_by_image_id[image_id][anno_index]
        return self.compute_3d_bbox_and_2d_projection(uniq_id)

    def _extract_rgbxyz_pcd(self, image_id: str):
        scene = self.scene_by_image_id[image_id]
        fx, fy, cx, cy, height, width = self.intrinsic_fetcher[image_id]

        Tyz = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]], dtype=np.float32)
        iTyz = np.linalg.inv(Tyz)
        # print(iTyz)
        # Tyz: (x, y, z) -> (x, z, -y)
        # iTyz: (x, y, z) -> (x, -z, y)

        color_raw = cv2.imread(str(scene.rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(scene.depth_path), cv2.IMREAD_UNCHANGED)
        color_np = np.asarray(color_raw, dtype=np.uint8)
        depth_np = np.asarray(depth_raw, dtype=np.uint16)
        mask_invalid = depth_np == 0

        pcd, m, rx, ry = convert_pcd(raw_depth=depth_np, extrinsics=scene.extrinsics, fx=fx, fy=fy, cx=cx, cy=cy)
        rgb = normalize_rgb(raw_rgb=color_np)
        xyzrgb = np.concatenate((pcd, rx[..., np.newaxis], ry[..., np.newaxis], rgb), axis=-1)
        # xyzrgb[..., 1] *= -1.
        # xyzrgb[..., 2] *= -1.

        # new_xyz = xyzrgb[m, :]
        # from scipy import interpolate
        #
        # new_rgb = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
        # xl, yl, rl, gl, bl = [], [], [], [], []
        # for i in range(new_xyz.shape[0]):
        #     X, Y, Z = new_xyz[i, 0], new_xyz[i, 1], new_xyz[i, 2]
        #     R, G, B = new_xyz[i, 5], new_xyz[i, 6], new_xyz[i, 7]
        #     xx = X / Z * fx + cx
        #     yy = Y / Z * fy + cy
        #     if 0 <= xx < width * 2 and 0 <= yy < height * 2:
        #         rr = int((R * 0.229 + 0.485) * 255)
        #         gg = int((G * 0.224 + 0.456) * 255)
        #         bb = int((B * 0.225 + 0.406) * 255)
        #         new_rgb[int(yy), int(xx), :] = rr, gg, bb
        # plt.imshow(new_rgb)
        # plt.show()

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(xyzrgb[..., :3][m, :])

        tiE = np.eye(4, dtype=np.float32)
        tiE[:3, :3] = np.linalg.inv(scene.extrinsics)
        if self.verbose:
            o3d_obj_list = []
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
            coord_frame.transform(iTyz)
            # coord_frame.transform(tiE)
            o3d_obj_list.append(coord_frame)
            o3d_obj_list.append(new_pcd)

        aabb_bbox_list = []
        for oid, bbox_3d in enumerate(scene.gt_3d_bbox):
            class_name = bbox_3d.class_name  # str
            centroid = bbox_3d.centroid[0]  # (3, )
            basis = bbox_3d.basis
            # basis = np.transpose(scene.extrinsics) @ bbox_3d.basis  # (3, 3)
            # basis = np.transpose(scene.extrinsics) @ bbox_3d.basis
            # print(bbox_3d.basis)
            # print(basis)
            # print()
            coeffs = bbox_3d.coeffs[0]  # (3, )
            line_mesh = create_line_mesh(centroid, basis, coeffs, aabb=self.aabb)
            line_mesh.transform(iTyz)
            # line_mesh.transform(tiE)
            # line_mesh.transform(tiE)
            # line_mesh.transform(tr_flip)
            if self.verbose:
                o3d_obj_list += line_mesh.cylinder_segments

            # # Testing center, size representation.
            # bbox_points = np.array(line_mesh.points)
            # p1 = np.min(bbox_points, axis=0)
            # p2 = np.max(bbox_points, axis=0)
            # bbox_points = np.concatenate(((p1 + p2) / 2., p2 - p1), axis=0)
            # if self.verbose:
            #     line_mesh2 = create_line_mesh_from_center_size(bbox_points)
            #     o3d_obj_list += line_mesh2.cylinder_segments
            # aabb_bbox_list.append((class_name, bbox_points.tolist()))

        if self.verbose:
            o3d.visualization.draw_geometries(o3d_obj_list)
        return xyzrgb, aabb_bbox_list, mask_invalid

    def compute_depth_mask(self, image_id: str):
        return np.asarray(cv2.imread(str(self.scene_by_image_id[image_id].depth_path), cv2.IMREAD_UNCHANGED),
                          dtype=np.uint16) > 0

    def extract_pcd_with_mask(self, image_id: str):
        scene = self.scene_by_image_id[image_id]
        fx, fy, cx, cy, height, width = self.intrinsic_fetcher[image_id]

        color_raw = cv2.imread(str(scene.rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(scene.depth_path), cv2.IMREAD_UNCHANGED)
        color_np = np.asarray(color_raw, dtype=np.uint8)
        depth_np = np.asarray(depth_raw, dtype=np.uint16)
        mask_valid = depth_np > 0

        pcd, m, rx, ry = convert_pcd(raw_depth=depth_np, extrinsics=scene.extrinsics, fx=fx, fy=fy, cx=cx, cy=cy)
        rgb = normalize_rgb(raw_rgb=color_np)
        xyzrgb = np.concatenate((pcd, rx[..., np.newaxis], ry[..., np.newaxis], rgb), axis=-1)
        xyzrgb[..., 1] *= -1
        xyzrgb[..., 2] *= -1

        return xyzrgb, mask_valid

    def compute_3d_bbox(self, uniq_id: str):
        image_id, object_id, anno_id = uniq_id.split('_')
        object_id = int(object_id)
        scene = self.scene_by_image_id[image_id]
        fx, fy, cx, cy, height, width = self.intrinsic_fetcher[image_id]

        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = scene.extrinsics
        Tyz = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]], dtype=np.float32)
        Rt = Tyz @ Rt

        color_raw = cv2.imread(str(scene.rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(scene.depth_path), cv2.IMREAD_UNCHANGED)
        rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(
            o3d.geometry.Image(color_raw),
            o3d.geometry.Image(depth_raw),
            convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=rgbd_image,
            intrinsic=o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy))
        # pcd.transform(Rt)

        o3d_obj_list = []
        o3d_obj_list.append(pcd)

        coord_frame_orig = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        coord_frame_revised = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        coord_frame_revised.transform(np.linalg.inv(Rt))

        o3d_obj_list.append(coord_frame_orig)
        o3d_obj_list.append(coord_frame_revised)

        for oid, bbox_3d in enumerate(scene.gt_3d_bbox):
            # class_name = bbox_3d.class_name[0]  # str
            centroid = bbox_3d.centroid[0]  # (3, )
            basis = bbox_3d.basis  # (3, 3)
            coeffs = bbox_3d.coeffs[0]  # (3, )

            if self.highlight and oid == object_id:
                color = [0, 1, 0]
                radius = 0.03
            else:
                color = [1, 0, 0]
                radius = 0.015

            line_mesh = create_line_mesh(centroid, basis, coeffs, aabb=self.aabb, color=color, radius=radius)
            for s in line_mesh.cylinder_segments:
                s.transform(np.linalg.inv(Rt))
            o3d_obj_list += line_mesh.cylinder_segments
        o3d.visualization.draw_geometries(o3d_obj_list)

    def compute_3d_bbox_and_2d_projection(self, uniq_id: str):
        image_id, object_id, anno_id = uniq_id.split('_')
        object_id = int(object_id)
        scene = self.scene_by_image_id[image_id]
        fx, fy, cx, cy, height, width = self.intrinsic_fetcher[image_id]

        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = scene.extrinsics
        Tyz = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]], dtype=np.float32)
        Rt = Tyz @ Rt
        iRt = np.linalg.inv(Rt)[:3, :3]

        color_raw = cv2.imread(str(scene.rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(scene.depth_path), cv2.IMREAD_UNCHANGED)
        rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(
            o3d.geometry.Image(color_raw),
            o3d.geometry.Image(depth_raw),
            convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=rgbd_image,
            intrinsic=o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy))
        pcd.transform(Rt)

        # pcd = o3d.geometry.PointCloud()
        # points = np.load('/home/junha/Downloads/{}_pcd.npy'.format(image_id))
        # mask = np.load('/home/junha/Downloads/{}_mask.npy'.format(image_id))
        # pcd.points = o3d.utility.Vector3dVector(points[..., :3][mask, :])

        o3d_obj_list = []
        o3d_obj_list.append(pcd)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        o3d_obj_list.append(coord_frame)

        for oid, bbox_3d in enumerate(scene.gt_3d_bbox):
            class_name = bbox_3d.class_name
            centroid = bbox_3d.centroid[0]  # (3, )
            basis = bbox_3d.basis  # (3, 3)
            coeffs = bbox_3d.coeffs[0]  # (3, )

            if self.highlight and oid == object_id:
                color = [0, 1, 0]
                radius = 0.03
            else:
                color = [1, 0, 0]
                radius = 0.015

            line_mesh = create_line_mesh(centroid, basis, coeffs, aabb=self.aabb, color=color, radius=radius)
            o3d_obj_list += line_mesh.cylinder_segments

            p1 = np.min(line_mesh.points, axis=0)
            p2 = np.max(line_mesh.points, axis=0)
            y = (p1[1] + p2[1]) * 0.5
            x1, x2 = p1[0], p2[0]
            z1, z2 = p1[2], p2[2]

            # Choose a plane with the mean y-value.
            # P1 +-----+ P2
            #    |     |
            # P4 +-----+ P3
            P1 = np.array((x1, y, z2)) @ np.transpose(iRt)
            P2 = np.array((x2, y, z2)) @ np.transpose(iRt)
            P3 = np.array((x2, y, z1)) @ np.transpose(iRt)
            P4 = np.array((x1, y, z1)) @ np.transpose(iRt)

            # Project the corner points into 2D plane.
            def project_2d(P, fx, fy, cx, cy):
                return P[0] / P[2] * fx + cx, P[1] / P[2] * fy + cy

            pp1 = project_2d(P1, fx, fy, cx, cy)
            pp2 = project_2d(P2, fx, fy, cx, cy)
            pp3 = project_2d(P3, fx, fy, cx, cy)
            pp4 = project_2d(P4, fx, fy, cx, cy)

            print(oid, class_name)
            print(bbox_3d)
            print(line_mesh.points)
            print(P1, P2, P3, P4)
            print(pp1, pp2, pp3, pp4)
            y1 = 0.5 * (pp1[1] + pp2[1])
            y2 = 0.5 * (pp3[1] + pp4[1])
            x1 = 0.5 * (pp1[0] + pp4[0])
            x2 = 0.5 * (pp2[0] + pp3[0])
            print(x1, x2, y1, y2)

            cv2.rectangle(img=color_raw,
                          pt1=(int(x1), int(y1)),
                          pt2=(int(x2), int(y2)),
                          color=(255, 0, 0),
                          thickness=2)
            cv2.putText(img=color_raw,
                        text=class_name,
                        org=(int(x1), int(y1)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(255, 0, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA)
        plt.imshow(color_raw)
        plt.show()

        o3d.visualization.draw_geometries(o3d_obj_list)

    def visualize_pcd_and_gt_bbox(self, uniq_id: str):
        image_id, object_id, anno_id = uniq_id.split('_')
        cls_aabb_list = self.cls_aabb_list[image_id]

        pcd = o3d.geometry.PointCloud()
        points = np.load('/home/junha/data/sunrefer/xyzrgb/{}.npy'.format(image_id))
        mask = np.load('/home/junha/data/sunrefer/xyzrgb/{}_mask.npy'.format(image_id))
        pcd.points = o3d.utility.Vector3dVector(points[..., :3][mask, :])

        o3d_obj_list = []
        o3d_obj_list.append(pcd)
        for cls, aabb in cls_aabb_list:
            line_mesh = create_line_mesh_from_center_size(np.array(aabb))
            o3d_obj_list += line_mesh.cylinder_segments
        o3d.visualization.draw_geometries(o3d_obj_list)


def create_xyzrgb_and_aabb():
    vis = PredictionVisualizer(
        aabb=True,
        highlight=True,
        apply_seg_out_mask=True,
        verbose=False,
        bbox_2d_ratio=1.0,
        pcd_th_ratio=0.1,
        mask_ratio=0.8)

    aabb_by_image_id = dict()
    for i in tqdm(range(1, 10336)):
        image_id = '{:06d}'.format(i)
        xyzrgb, aabb_list = vis.extract_rgbxyz_pcd(image_id)
        np.save(str(fetch_xyzrgb_pcd_path(image_id)), xyzrgb)
        aabb_by_image_id[image_id] = aabb_list
    with open(str(fetch_xyzrgb_bbox_path()), 'w') as file:
        json.dump(aabb_by_image_id, file, indent=4)


def visualize_annotation(image_id: str, anno_idx: int):
    vis = PredictionVisualizer(
        aabb=True,
        highlight=True,
        apply_seg_out_mask=True,
        verbose=True,
        bbox_2d_ratio=1.0,
        pcd_th_ratio=0,
        mask_ratio=0.8)
    vis.compute_3d_bbox_by_image_id(image_id, anno_idx)


def test():
    vis = PredictionVisualizer(
        aabb=False,
        highlight=True,
        apply_seg_out_mask=True,
        verbose=True,
        bbox_2d_ratio=1.0,
        pcd_th_ratio=0,
        mask_ratio=0.8)

    image_id = '002920'
    pcd, mask = vis.extract_pcd_with_mask(image_id)
    tmp_dir = Path.home() / 'Downloads'
    torch.save({'pcd': pcd, 'mask': mask}, str(tmp_dir / '{}.pt'.format(image_id)))
    # np.save(str(tmp_dir / '{}_pcd.npy'.format(image_id)), pcd)
    # np.save(str(tmp_dir / '{}_mask.npy'.format(image_id)), mask)
    print(pcd.shape)
    print(mask.shape)


def create_aabb_from_bbox_3d(bbox_3d, center_size: bool):
    centroid = bbox_3d.centroid[0]  # (3, )
    basis = bbox_3d.basis  # (3, 3)
    coeffs = bbox_3d.coeffs[0]  # (3, )
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
    cx, cy, cz, sx, sy, sz = aabb_from_oriented_bbox(np.array(pl))

    if center_size:
        return cx + centroid[0], cy + centroid[1], cz + centroid[2], sx, sy, sz
    else:
        return [[cx + 0.5 * dx * sx + centroid[0],
                 cy + 0.5 * dy * sy + centroid[1],
                 cz + 0.5 * dz * sz + centroid[2]] for dx, dy, dz in cl]


def fetch_2d_bbox_from_3d(bbox_3d, iRt, fx, fy, cx, cy):
    cx_, cy_, cz_, sx_, sy_, sz_ = create_aabb_from_bbox_3d(bbox_3d, center_size=True)
    y_ = cy_
    x1_, x2_ = cx_ - 0.5 * sx_, cx_ + 0.5 * sx_
    z1_, z2_ = cz_ - 0.5 * sz_, cz_ + 0.5 * sz_

    # Choose a plane with the mean y-value.
    # P1 +-----+ P2
    #    |     |
    # P4 +-----+ P3
    P1 = np.array((x1_, y_, z2_)) @ np.transpose(iRt)
    P2 = np.array((x2_, y_, z2_)) @ np.transpose(iRt)
    P3 = np.array((x2_, y_, z1_)) @ np.transpose(iRt)
    P4 = np.array((x1_, y_, z1_)) @ np.transpose(iRt)

    # Project the corner points into 2D plane.
    def project_2d(P, fx, fy, cx, cy):
        return P[0] / P[2] * fx + cx, P[1] / P[2] * fy + cy

    pp1 = project_2d(P1, fx, fy, cx, cy)
    pp2 = project_2d(P2, fx, fy, cx, cy)
    pp3 = project_2d(P3, fx, fy, cx, cy)
    pp4 = project_2d(P4, fx, fy, cx, cy)

    y1 = 0.5 * (pp1[1] + pp2[1])
    y2 = 0.5 * (pp3[1] + pp4[1])
    x1 = 0.5 * (pp1[0] + pp4[0])
    x2 = 0.5 * (pp2[0] + pp3[0])
    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    # class_name = bbox_3d.class_name
    # cv2.rectangle(img=color_raw,
    #               pt1=(int(x1), int(y1)),
    #               pt2=(int(x2), int(y2)),
    #               color=(255, 0, 0),
    #               thickness=2)
    # cv2.putText(img=color_raw,
    #             text=class_name,
    #             org=(int(x1), int(y1)),
    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #             fontScale=1,
    #             color=(255, 0, 0),
    #             thickness=2,
    #             lineType=cv2.LINE_AA)
    return x, y, w, h


def compute_iou_xywh(v1, v2):
    x1, y1, w1, h1 = v1
    x2, y2, w2, h2 = v2
    a1, a2 = w1 * h1, w2 * h2

    Ax1, Ax2, Ay1, Ay2 = x1, x1 + w1, y1, y1 + h1
    Bx1, Bx2, By1, By2 = x2, x2 + w2, y2, y2 + h2
    Cx1, Cx2 = max(Ax1, Bx1), min(Ax2, Bx2)
    Cy1, Cy2 = max(Ay1, By1), min(Ay2, By2)

    if Cx1 >= Cx2 or Cy1 >= Cy2:
        return 0.

    ai = (Cx2 - Cx1) * (Cy2 - Cy1)
    au = a1 + a2 - ai
    if au <= 1e-5:
        return 0.
    return ai / au


def iou_stats_from_list(iou_list):
    if not iou_list:
        print('got empty iou list')
    else:
        iou = np.array(iou_list).reshape(-1, )
        iou_25 = np.sum(iou > 0.25) / iou.shape[0] * 100
        iou_50 = np.sum(iou > 0.5) / iou.shape[0] * 100
        print('mean iou: {:5.3f}'.format(np.mean(iou) * 100))
        print('acc@0.25: {:5.3f}'.format(iou_25))
        print('acc@0.5 : {:5.3f}'.format(iou_50))


def compute_3d_to_2d_projection():
    scene_by_image_id = fetch_scene_object_by_image_id('v2_3d')
    intrinsic_fetcher = IntrinsicFetcher()

    with open(str(fetch_sunrefer_anno_path()), 'r') as file:
        anno_dict = json.load(file)

    test_iou = []
    train_iou = []
    for image_object_id, annotation in anno_dict.items():
        image_id, object_id = image_object_id.split('_')
        _, _, _, _, h, w = intrinsic_fetcher[image_id]
        train = int(image_id) >= 5051
        object_id = int(object_id)
        scene = scene_by_image_id[image_id]
        gt_bbox_2d = annotation['bbox2d']
        pr_bbox_2d = scene.project_2d_bbox_from_3d_bbox(object_id, w=w, h=h)
        iou = compute_iou_xywh(gt_bbox_2d, pr_bbox_2d)
        if train:
            train_iou.append(iou)
        else:
            test_iou.append(iou)

    print('TRAIN')
    iou_stats_from_list(train_iou)
    print('TEST')
    iou_stats_from_list(test_iou)

    # for uniq_id in ['000001_4_0']:
    #     image_id, object_id, anno_id = uniq_id.split('_')
    #     object_id = int(object_id)
    #     scene = scene_by_image_id[image_id]
    #     fx, fy, cx, cy, height, width = intrinsic_fetcher[image_id]
    #     bbox_2d = fetch_2d_bbox_from_3d(scene.gt_3d_bbox[object_id],
    #                                     compute_inverse_rt(scene.extrinsics), fx, fy, cx, cy)
    #     print(bbox_2d)


if __name__ == '__main__':
    # test()
    # visualize_annotation('000001', 0)
    # compute_3d_to_2d_projection()
    # compute_average_iou(aabb=True,
    #                     use_seg_out_mask=True,
    #                     bbox_2d_ratio=1.0,
    #                     pcd_th_ratio=0.1,
    #                     mask_ratio=0.8)
    # visualize_annotation(image_id='002920', anno_idx=0)
    # test()
    vis = PredictionVisualizer(
        aabb=False,
        highlight=True,
        apply_seg_out_mask=True,
        verbose=True,
        bbox_2d_ratio=1.0,
        pcd_th_ratio=0,
        mask_ratio=0.8)
    image_id = '002920'
    vis.visualize_pcd_and_gt_bbox('{}_1_0'.format(image_id))
    # for i in tqdm(range():
    #     image_id = '{:06d}'.format(i)
        # mask = vis.compute_depth_mask(image_id)
        # pcd_dir = Path.home() / 'data/sunrefer/xyzrgb'
        # np.save(str(tmp_dir / '{}_pcd.npy'.format(image_id)), pcd)
        # np.save(str(pcd_dir / '{}_mask.npy'.format(image_id)), mask)
        # torch.save({'pcd': pcd, 'mask': mask}, str(tmp_dir / '{}.pt'.format(image_id)))
        # vis.visualize_pcd_and_gt_bbox('{}_1_0'.format(image_id))
