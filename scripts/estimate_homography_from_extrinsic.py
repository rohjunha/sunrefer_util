import json
from multiprocessing import Pool
from pathlib import Path
from shutil import copyfile
from typing import List, Dict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from scripts.compute_iou import create_line_mesh_from_center_size
from utils.intrinsic_fetcher import IntrinsicFetcher
from utils.meta_io import fetch_scene_object_by_image_id


def convert_depth_float_from_uint8(raw_depth: np.ndarray):
    assert raw_depth.dtype == np.uint16
    depth = np.bitwise_or(np.right_shift(raw_depth, 3), np.left_shift(raw_depth, 13)).astype(np.float32) / 1000.
    depth[depth > 7.0] = 0.
    return depth


def unproject(float_depth, fx, fy, cx, cy):
    mask = float_depth > 0.
    mx = np.linspace(0, float_depth.shape[1] - 1, float_depth.shape[1], dtype=np.float32)
    my = np.linspace(0, float_depth.shape[0] - 1, float_depth.shape[0], dtype=np.float32)
    rx = np.linspace(0., 1., float_depth.shape[1], dtype=np.float32)
    ry = np.linspace(0., 1., float_depth.shape[0], dtype=np.float32)
    rx, ry = np.meshgrid(rx, ry)
    xx, yy = np.meshgrid(mx, my)
    xx = (xx - cx) * float_depth / fx
    yy = (yy - cy) * float_depth / fy
    pcd = np.stack((xx, yy, float_depth), axis=-1)
    return pcd, mask, rx, ry


def transform(pcd, E):
    return pcd @ np.transpose(E)


def normalize_rgb(raw_rgb):
    float_rgb = np.array(raw_rgb).astype(np.float32) / 255.
    float_rgb[..., 0] = (float_rgb[..., 0] - 0.485) / 0.229
    float_rgb[..., 1] = (float_rgb[..., 1] - 0.456) / 0.224
    float_rgb[..., 2] = (float_rgb[..., 2] - 0.406) / 0.225
    return float_rgb


def denormalize_rgb(float_rgb):
    r = (float_rgb[..., 0] * 0.229 + 0.485) * 255.
    g = (float_rgb[..., 1] * 0.224 + 0.456) * 255.
    b = (float_rgb[..., 2] * 0.225 + 0.406) * 255.
    return np.stack((r, g, b), axis=-1).astype(np.uint8)


def fetch_raw_rgb(image_id: str):
    raw_bgr = cv2.imread('/home/junha/data/sunrefer/rgb/{}.jpg'.format(image_id), cv2.IMREAD_COLOR)
    return cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)


def fetch_raw_depth(image_id: str):
    return cv2.imread('/home/junha/data/sunrefer/raw_depth/{}.png'.format(image_id), cv2.IMREAD_UNCHANGED)


class TransformManager:
    def __init__(self):
        self.meta_dict = torch.load('/home/junha/data/sunrefer/meta.pt')
        self.intrinsic_fetcher = IntrinsicFetcher()
        self.scene_by_image_id = fetch_scene_object_by_image_id('v2_3d')
        self.aabb_dict = json.load(open('/home/junha/data/sunrefer/xyzrgb/aabb.json', 'r'))

    def fetch_original_pcd(self, image_id: str):
        fx, fy, cx, cy, h, w = self.intrinsic_fetcher[image_id]
        E = np.array(self.meta_dict[image_id]['E'], dtype=np.float32)
        raw_depth = fetch_raw_depth(image_id)
        float_depth = convert_depth_float_from_uint8(raw_depth)
        pcd, mask, rx, ry = unproject(float_depth, fx, fy, cx, cy)
        return transform(pcd, E), mask, rx, ry

    def fetch_warp_pcd(self, image_id: str):
        fx, fy, cx, cy, h, w = self.intrinsic_fetcher[image_id]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        E = np.array(self.meta_dict[image_id]['E'], dtype=np.float32)
        T = (K @ E @ np.linalg.inv(K)).astype(dtype=np.float32)
        T = T / T[2, 2]

        p1 = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]], dtype=np.float32)
        p2 = T @ p1
        p2 = np.divide(p2[:2, :], np.tile(p2[-1, :], (2, 1)))

        # offset_x = -np.min(p2[0, :])
        # offset_y = -np.min(p2[1, :])
        # p2[0, :] += offset_x
        # p2[1, :] += offset_y
        width = min(2000, int(round(np.max(p2[0, :]))))
        height = min(2000, int(round(np.max(p2[1, :]))))

        # translate the homography w.r.t the offset.
        # T[0, 2] += offset_x
        # T[1, 2] += offset_y
        # cx += offset_x
        # cy += offset_y

        intrinsic = {'width': width, 'height': height, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                     'H': np.reshape(T, -1).tolist()}

        raw_depth = fetch_raw_depth(image_id)
        float_depth = convert_depth_float_from_uint8(raw_depth)
        mask = float_depth > 0

        depth_mask_orig = np.concatenate((float_depth[..., np.newaxis], mask[..., np.newaxis]), axis=-1)
        depth_mask_warp = cv2.warpPerspective(depth_mask_orig, T, (width, height))

        return unproject(depth_mask_warp[..., 0], fx, fy, cx, cy)

    def fetch_pixel_pcd(self, image_id: str):
        fx, fy, cx, cy, h, w = self.intrinsic_fetcher[image_id]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        E = np.array(self.meta_dict[image_id]['E'], dtype=np.float32)
        T = (K @ E @ np.linalg.inv(K)).astype(dtype=np.float32)
        T = T / T[2, 2]

        p1 = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]], dtype=np.float32)
        p2 = T @ p1
        p2 = np.divide(p2[:2, :], np.tile(p2[-1, :], (2, 1)))

        # offset_x = round(-np.min(p2[0, :])) if np.min(p2[0, :]) < 0 else 0
        # offset_y = round(-np.min(p2[1, :])) if np.min(p2[1, :]) < 0 else 0
        offset_x = -np.min(p2[0, :])
        offset_y = -np.min(p2[1, :])
        p2[0, :] += offset_x
        p2[1, :] += offset_y
        width = min(2000, int(round(np.max(p2[0, :]))))
        height = min(2000, int(round(np.max(p2[1, :]))))
        cx += offset_x
        cy += offset_y

        raw_rgb = normalize_rgb(fetch_raw_rgb(image_id))
        pcd_orig, mask_orig, _, _ = self.fetch_original_pcd(image_id)

        sampled_mask = np.zeros((height, width), dtype=bool)
        sampled_depth = np.zeros((height, width), dtype=np.float32)
        sampled_rgb = np.zeros((height, width, 3), dtype=np.float32)

        xx_ = fx * pcd_orig[..., 0] / pcd_orig[..., 2] + cx
        yy_ = fy * pcd_orig[..., 1] / pcd_orig[..., 2] + cy
        mm = np.logical_and(~np.isnan(xx_), ~np.isnan(yy_))
        xx = np.round(xx_).astype(dtype=np.int32)
        yy = np.round(yy_).astype(dtype=np.int32)
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, pcd_orig.shape[1] - 1, pcd_orig.shape[1]),
            np.linspace(0, pcd_orig.shape[0] - 1, pcd_orig.shape[0]))
        grid_x = grid_x.astype(dtype=np.int32)
        grid_y = grid_y.astype(dtype=np.int32)
        xx = xx[mm]
        yy = yy[mm]
        grid_x = grid_x[mm]
        grid_y = grid_y[mm]
        mm = (0 <= xx) & (xx < width) & (0 <= yy) & (yy < height)
        xx = xx[mm]
        yy = yy[mm]
        grid_x = grid_x[mm]
        grid_y = grid_y[mm]

        sampled_mask[yy, xx] = True
        sampled_depth[yy, xx] = pcd_orig[grid_y, grid_x, 2]
        sampled_rgb[yy, xx, :] = raw_rgb[grid_y, grid_x, :]
        pcd, mask, rx, ry = unproject(sampled_depth, fx, fy, cx, cy)

        mask_ = mask[..., np.newaxis].astype(dtype=np.float32)
        rx_ = rx[..., np.newaxis]
        ry_ = ry[..., np.newaxis]
        data = np.concatenate((pcd, rx_, ry_, sampled_rgb, mask_), axis=-1)
        data[~mask, :] = 0.

        intrinsic = {'width': width, 'height': height, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                     'offset_x': offset_x, 'offset_y': offset_y, 'H': np.reshape(T, -1).tolist()}
        return data, intrinsic

    def fetch_pixel_depth(self, image_id: str):
        fx, fy, cx, cy, h, w = self.intrinsic_fetcher[image_id]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        E = np.array(self.meta_dict[image_id]['E'], dtype=np.float32)
        T = (K @ E @ np.linalg.inv(K)).astype(dtype=np.float32)
        T = T / T[2, 2]

        p1 = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]], dtype=np.float32)
        p2 = T @ p1
        p2 = np.divide(p2[:2, :], np.tile(p2[-1, :], (2, 1)))

        p2x = p2[0, np.abs(p2[0, :]) < 5000]
        p2y = p2[1, np.abs(p2[1, :]) < 5000]
        offset_x = -np.min(p2x)
        offset_y = -np.min(p2y)
        p2x += offset_x
        p2y += offset_y
        width = min(3000, int(round(np.max(p2x))))
        height = min(3000, int(round(np.max(p2y))))

        if offset_x > 2000 or offset_y > 2000:
            data = np.zeros((h, w, 7), dtype=np.float32)
            intrinsic = {'width': w, 'height': h, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                         'offset_x': 0., 'offset_y': 0., 'H': np.eye(3, dtype=np.float32).tolist()}
            return data, intrinsic

        cx += offset_x
        cy += offset_y

        raw_rgb = normalize_rgb(fetch_raw_rgb(image_id))
        pcd_orig, mask_orig, _, _ = self.fetch_original_pcd(image_id)

        sampled_mask = np.zeros((height, width), dtype=bool)
        sampled_depth = np.zeros((height, width), dtype=np.float32)
        sampled_rgb = np.zeros((height, width, 3), dtype=np.float32)

        xx_ = fx * pcd_orig[..., 0] / pcd_orig[..., 2] + cx
        yy_ = fy * pcd_orig[..., 1] / pcd_orig[..., 2] + cy
        mm = np.logical_and(~np.isnan(xx_), ~np.isnan(yy_))
        xx = np.round(xx_).astype(dtype=np.int32)
        yy = np.round(yy_).astype(dtype=np.int32)
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, pcd_orig.shape[1] - 1, pcd_orig.shape[1]),
            np.linspace(0, pcd_orig.shape[0] - 1, pcd_orig.shape[0]))
        grid_x = grid_x.astype(dtype=np.int32)
        grid_y = grid_y.astype(dtype=np.int32)
        xx = xx[mm]
        yy = yy[mm]
        grid_x = grid_x[mm]
        grid_y = grid_y[mm]
        mm = (0 <= xx) & (xx < width) & (0 <= yy) & (yy < height)
        xx = xx[mm]
        yy = yy[mm]
        grid_x = grid_x[mm]
        grid_y = grid_y[mm]

        sampled_mask[yy, xx] = True
        sampled_depth[yy, xx] = pcd_orig[grid_y, grid_x, 2]
        sampled_rgb[yy, xx, :] = raw_rgb[grid_y, grid_x, :]
        rx = np.linspace(0., 1., sampled_depth.shape[1], dtype=np.float32)
        ry = np.linspace(0., 1., sampled_depth.shape[0], dtype=np.float32)
        rx, ry = np.meshgrid(rx, ry)

        depth_ = sampled_depth[..., np.newaxis]
        mask_ = sampled_mask[..., np.newaxis].astype(dtype=np.float32)
        rx_ = rx[..., np.newaxis]
        ry_ = ry[..., np.newaxis]
        data = np.concatenate((depth_, rx_, ry_, sampled_rgb, mask_), axis=-1)
        data[~sampled_mask, :] = 0.

        intrinsic = {'width': width, 'height': height, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                     'offset_x': offset_x, 'offset_y': offset_y, 'H': np.reshape(T, -1).tolist()}
        return data, intrinsic

    def fetch_saved_pcd(self, image_id: str):
        pcd = np.load(str(Path.home() / 'data/sunrefer/xyzrgb_extrinsic/{}.npy'.format(image_id)))
        for i in range(3):
            print(i, np.min(pcd[..., i]), np.max(pcd[..., i]))
        return pcd[..., :3]

    def visualize_pcds(self, image_id: str):
        import open3d as o3d

        pcd1, mask1, _, _ = self.fetch_original_pcd(image_id)
        pcd2, mask2, _, _ = self.fetch_warp_pcd(image_id)
        # pcd3_, intrinsic = self.fetch_pixel_pcd(image_id)  # (x, y, z, rx, ry, r, g, b, mask)
        # pcd3, mask3 = pcd3_[..., :3], pcd3_[..., -1] > 1e-5

        data3, intrinsic = self.fetch_pixel_depth(image_id)
        print(np.sum(data3))
        print(intrinsic)
        pcd3, mask3, _, _ = unproject(data3[..., 0], intrinsic['fx'], intrinsic['fy'], intrinsic['cx'], intrinsic['cy'])

        pcd4 = self.fetch_saved_pcd(image_id)

        # depth = np.right_shift((pcd3[..., -1] * 1000).astype(dtype=np.uint16), 3)
        # color = denormalize_rgb(pcd3_[..., 5:8])
        # color[~mask3, :] = 0
        # o3d.io.write_image('/home/junha/Downloads/depth{}.png'.format(image_id), o3d.geometry.Image(depth))
        # o3d.io.write_image('/home/junha/Downloads/color{}.png'.format(image_id), o3d.geometry.Image(color))

        pt1 = o3d.geometry.PointCloud()
        pt2 = o3d.geometry.PointCloud()
        pt3 = o3d.geometry.PointCloud()
        pt4 = o3d.geometry.PointCloud()
        pt1.points = o3d.utility.Vector3dVector(pcd1[mask1, :])
        pt2.points = o3d.utility.Vector3dVector(pcd2[mask2, :])
        pt3.points = o3d.utility.Vector3dVector(pcd3[mask3, :])
        pt4.points = o3d.utility.Vector3dVector(np.reshape(pcd4, (-1, 3)))
        pt1.paint_uniform_color((1, 0, 0))  # red
        pt2.paint_uniform_color((0, 1, 0))  # green
        pt3.paint_uniform_color((0, 0, 1))  # blue
        pt4.paint_uniform_color((0, 1, 0))  # green

        obj_list = [pt1, pt3, pt4]
        for i, (_, aabb) in enumerate(self.aabb_dict[image_id]):
            cx_, cy_, cz_, sx, sy, sz = aabb
            line_mesh = create_line_mesh_from_center_size(np.array([cx_, -cy_, -cz_, sx, sy, sz]))
            obj_list += line_mesh.cylinder_segments
        o3d.visualization.draw_geometries(obj_list)


meta_dict = torch.load('/home/junha/data/sunrefer/meta.pt')
intrinsic_fetcher = IntrinsicFetcher()
scene_by_image_id = fetch_scene_object_by_image_id('v2_3d')
aabb_dict = json.load(open('/home/junha/data/sunrefer/xyzrgb/aabb.json', 'r'))
out_dir = Path.home() / 'data/sunrefer/xyzrgb_concise'


def fetch_original_pcd(image_id: str):
    fx, fy, cx, cy, h, w = intrinsic_fetcher[image_id]
    E = np.array(meta_dict[image_id]['E'], dtype=np.float32)
    raw_depth = fetch_raw_depth(image_id)
    float_depth = convert_depth_float_from_uint8(raw_depth)
    pcd, mask, rx, ry = unproject(float_depth, fx, fy, cx, cy)
    return transform(pcd, E), mask, rx, ry


def fetch_pixel_depth(image_id: str):
    out_rgb_path = out_dir / '{}.jpg'.format(image_id)
    out_depth_path = out_dir / '{}.png'.format(image_id)
    out_intrinsic_path = out_dir / '{}.json'.format(image_id)

    fx, fy, cx, cy, h, w = intrinsic_fetcher[image_id]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    E = np.array(meta_dict[image_id]['E'], dtype=np.float32)
    T = (K @ E @ np.linalg.inv(K)).astype(dtype=np.float32)
    T = T / T[2, 2]

    p1 = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]], dtype=np.float32)
    p2 = T @ p1
    p2 = np.divide(p2[:2, :], np.tile(p2[-1, :], (2, 1)))

    p2x = p2[0, np.abs(p2[0, :]) < 5000]
    p2y = p2[1, np.abs(p2[1, :]) < 5000]
    offset_x = -np.min(p2x)
    offset_y = -np.min(p2y)
    p2x += offset_x
    p2y += offset_y
    width = min(3000, int(round(np.max(p2x))))
    height = min(3000, int(round(np.max(p2y))))

    if offset_x > 2000 or offset_y > 2000:
        out_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        out_depth = np.zeros((h, w), dtype=np.uint16)
        out_intrinsic = {'width': w, 'height': h, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                         'offset_x': 0., 'offset_y': 0., 'H': np.eye(3, dtype=np.float32).tolist()}
        cv2.imwrite(str(out_rgb_path), out_rgb)
        cv2.imwrite(str(out_depth_path), out_depth)
        with open(str(out_intrinsic_path), 'w') as file:
            json.dump(out_intrinsic, file)

    else:
        cx += offset_x
        cy += offset_y

        raw_rgb = fetch_raw_rgb(image_id)
        pcd_orig, mask_orig, _, _ = fetch_original_pcd(image_id)

        sampled_mask = np.zeros((height, width), dtype=bool)
        sampled_depth = np.zeros((height, width), dtype=np.float32)
        out_rgb = np.zeros((height, width, 3), dtype=np.uint8)

        xx_ = fx * pcd_orig[..., 0] / pcd_orig[..., 2] + cx
        yy_ = fy * pcd_orig[..., 1] / pcd_orig[..., 2] + cy
        mm = np.logical_and(~np.isnan(xx_), ~np.isnan(yy_))
        xx = np.round(xx_).astype(dtype=np.int32)
        yy = np.round(yy_).astype(dtype=np.int32)
        grid_x, grid_y = np.meshgrid(
            np.linspace(0, pcd_orig.shape[1] - 1, pcd_orig.shape[1]),
            np.linspace(0, pcd_orig.shape[0] - 1, pcd_orig.shape[0]))
        grid_x = grid_x.astype(dtype=np.int32)
        grid_y = grid_y.astype(dtype=np.int32)
        xx = xx[mm]
        yy = yy[mm]
        grid_x = grid_x[mm]
        grid_y = grid_y[mm]
        mm = (0 <= xx) & (xx < width) & (0 <= yy) & (yy < height)
        xx = xx[mm]
        yy = yy[mm]
        grid_x = grid_x[mm]
        grid_y = grid_y[mm]

        sampled_mask[yy, xx] = True
        sampled_depth[yy, xx] = pcd_orig[grid_y, grid_x, 2]
        sampled_depth[~sampled_mask] = 0.
        out_depth = np.left_shift((sampled_depth * 1000.).astype(dtype=np.uint16), 3)
        cv2.imwrite(str(out_depth_path), out_depth)

        out_rgb[yy, xx, :] = raw_rgb[grid_y, grid_x, :]
        out_rgb[~sampled_mask, :] = 0
        out_rgb = cv2.cvtColor(out_rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(out_rgb_path), out_rgb)

        out_intrinsic = {
            'width': int(width),
            'height': int(height),
            'fx': float(fx),
            'fy': float(fy),
            'cx': float(cx),
            'cy': float(cy),
            'offset_x': float(offset_x),
            'offset_y': float(offset_y),
            'H': np.reshape(T, -1).tolist()}
        with open(str(out_intrinsic_path), 'w') as file:
            json.dump(out_intrinsic, file)


def visualize_pixel_depth(image_id: str):
    in_rgb_path = out_dir / '{}.jpg'.format(image_id)
    in_depth_path = out_dir / '{}.png'.format(image_id)
    in_intrinsic_path = out_dir / '{}.json'.format(image_id)
    zz = convert_depth_float_from_uint8(cv2.imread(str(in_depth_path), cv2.IMREAD_UNCHANGED))
    with open(str(in_intrinsic_path), 'r') as file:
        intrinsic = json.load(file)
    xx = np.linspace(0, zz.shape[1] - 1, zz.shape[1], dtype=np.float32)
    yy = np.linspace(0, zz.shape[0] - 1, zz.shape[0], dtype=np.float32)
    xx, yy = np.meshgrid(xx, yy)
    xx = (xx - intrinsic['cx']) * zz / intrinsic['fx']
    yy = (yy - intrinsic['cy']) * zz / intrinsic['fy']
    xyz = np.stack((xx, yy, zz), axis=-1)
    mask = zz > 0

    # let us crop the image: (2, 101), (222, 665)
    use_crop = False
    if use_crop:
        mask_crop = mask[101:666, 2:223]
        xyz_crop = xyz[101:666, 2:223, :]
        pts = xyz_crop[mask_crop, :]
    else:
        pts = xyz[mask, :]

    import open3d as o3d
    obj_list = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    obj_list.append(pcd)
    for _, v in aabb_dict[image_id]:
        bbox = create_line_mesh_from_center_size(np.array([v[0], -v[1], -v[2], v[3], v[4], v[5]]))
        obj_list += bbox.cylinder_segments
    o3d.visualization.draw_geometries(obj_list)


def visualize_with_gt_box(image_id):
    in_warp_rgb_path = out_dir / '{}.jpg'.format(image_id)
    in_orig_rgb_path = '/home/junha/data/sunrefer/rgb/{}.jpg'.format(image_id)
    in_depth_path = out_dir / '{}.png'.format(image_id)
    in_intrinsic_path = out_dir / '{}.json'.format(image_id)

    warp_rgb = cv2.imread(str(in_warp_rgb_path), cv2.IMREAD_COLOR)
    orig_rgb = cv2.imread(str(in_orig_rgb_path), cv2.IMREAD_COLOR)
    intrinsic = json.load(open(str(in_intrinsic_path), 'r'))
    offset_x = intrinsic['offset_x']
    offset_y = intrinsic['offset_y']

    H = np.reshape(np.array(intrinsic['H'], dtype=np.float32), (3, 3))
    print(H)

    obj_list = meta_dict[image_id]['obj_2d']
    for obj_item in obj_list:
        x1, y1, w, h = obj_item['bbox_2d']
        cv2.rectangle(orig_rgb, (x1, y1), (x1 + w, y1 + h), color=(0, 0, 255), thickness=2)

        x2, y2 = x1 + w, y1 + h

        p1 = np.array([[x1, x2, x2, x1], [y1, y1, y2, y2], [1, 1, 1, 1]], dtype=np.float32)
        p2 = H @ p1
        p2 = np.divide(p2[:2, :], np.tile(p2[-1, :], (2, 1)))
        pts = np.transpose(np.round(p2).astype(dtype=np.int32))
        pts = np.reshape(pts, (-1, 1, 2))
        pts[..., 0] += int(offset_x)
        pts[..., 1] += int(offset_y)
        cv2.polylines(warp_rgb, [pts], isClosed=True, color=(0, 0, 255), thickness=2)


        x1 = int(0.5 * (p2[0, 0] + p2[0, 3]) + offset_x)
        x2 = int(0.5 * (p2[0, 1] + p2[0, 2]) + offset_x)
        y1 = int(0.5 * (p2[1, 0] + p2[1, 1]) + offset_y)
        y2 = int(0.5 * (p2[1, 2] + p2[1, 3]) + offset_y)

        x1 = int(min(warp_rgb.shape[1] - 1, max(0, x1)))
        x2 = int(min(warp_rgb.shape[1] - 1, max(x1, x2)))
        y1 = int(min(warp_rgb.shape[0] - 1, max(0, y1)))
        y2 = int(min(warp_rgb.shape[0] - 1, max(y1, y2)))

        cv2.rectangle(warp_rgb, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    cv2.imwrite('/home/junha/Downloads/{}_orig.jpg'.format(image_id), orig_rgb)
    cv2.imwrite('/home/junha/Downloads/{}_warp.jpg'.format(image_id), warp_rgb)

    # zz = convert_depth_float_from_uint8(cv2.imread(str(in_depth_path), cv2.IMREAD_UNCHANGED))
    # with open(str(in_intrinsic_path), 'r') as file:
    #     intrinsic = json.load(file)
    # xx = np.linspace(0, zz.shape[1] - 1, zz.shape[1], dtype=np.float32)
    # yy = np.linspace(0, zz.shape[0] - 1, zz.shape[0], dtype=np.float32)
    # xx, yy = np.meshgrid(xx, yy)
    # xx = (xx - intrinsic['cx']) * zz / intrinsic['fx']
    # yy = (yy - intrinsic['cy']) * zz / intrinsic['fy']
    # xyz = np.stack((xx, yy, zz), axis=-1)
    # mask = zz > 0

    # import open3d as o3d
    # obj_list = []
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts)
    # obj_list.append(pcd)
    # for _, v in aabb_dict[image_id]:
    #     bbox = create_line_mesh_from_center_size(np.array([v[0], -v[1], -v[2], v[3], v[4], v[5]]))
    #     obj_list += bbox.cylinder_segments
    # o3d.visualization.draw_geometries(obj_list)


def fetch_warped_bbox(orig_bbox, intrinsic):
    width = intrinsic['width']
    height = intrinsic['height']
    offset_x = intrinsic['offset_x']
    offset_y = intrinsic['offset_y']
    H = np.reshape(np.array(intrinsic['H'], dtype=np.float32), (3, 3))

    x1, y1, w, h = orig_bbox
    x2, y2 = x1 + w, y1 + h

    p1 = np.array([[x1, x2, x2, x1], [y1, y1, y2, y2], [1, 1, 1, 1]], dtype=np.float32)
    p2 = H @ p1
    p2 = np.divide(p2[:2, :], np.tile(p2[-1, :], (2, 1)))

    x1 = int(0.5 * (p2[0, 0] + p2[0, 3]) + offset_x)
    x2 = int(0.5 * (p2[0, 1] + p2[0, 2]) + offset_x)
    y1 = int(0.5 * (p2[1, 0] + p2[1, 1]) + offset_y)
    y2 = int(0.5 * (p2[1, 2] + p2[1, 3]) + offset_y)

    x1 = int(min(width - 1, max(0, x1)))
    x2 = int(min(width - 1, max(x1, x2)))
    y1 = int(min(height - 1, max(0, y1)))
    y2 = int(min(height - 1, max(y1, y2)))
    return x1, y1, x2 - x1, y2 - y1


def visualize_with_gt_box(image_id):
    in_warp_rgb_path = out_dir / '{}.jpg'.format(image_id)
    in_orig_rgb_path = '/home/junha/data/sunrefer/rgb/{}.jpg'.format(image_id)
    in_depth_path = out_dir / '{}.png'.format(image_id)
    in_intrinsic_path = out_dir / '{}.json'.format(image_id)

    warp_rgb = cv2.imread(str(in_warp_rgb_path), cv2.IMREAD_COLOR)
    orig_rgb = cv2.imread(str(in_orig_rgb_path), cv2.IMREAD_COLOR)
    intrinsic = json.load(open(str(in_intrinsic_path), 'r'))

    width = intrinsic['width']
    height = intrinsic['height']
    offset_x = intrinsic['offset_x']
    offset_y = intrinsic['offset_y']
    H = np.reshape(np.array(intrinsic['H'], dtype=np.float32), (3, 3))

    obj_list = meta_dict[image_id]['obj_2d']
    for obj_item in obj_list:
        # x1, y1, w, h = fetch_warped_bbox(obj_item['bbox_2d'], intrinsic)
        x1, y1, w, h = obj_item['bbox_2d']
        x2, y2 = x1 + w, y1 + h

        p1 = np.array([[x1, x2, x2, x1], [y1, y1, y2, y2], [1, 1, 1, 1]], dtype=np.float32)
        p2 = H @ p1
        p2 = np.divide(p2[:2, :], np.tile(p2[-1, :], (2, 1)))

        pts = np.transpose(np.round(p2).astype(dtype=np.int32))
        pts = np.reshape(pts, (-1, 1, 2))
        pts[..., 0] += int(offset_x)
        pts[..., 1] += int(offset_y)
        cv2.polylines(warp_rgb, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        x1 = int(0.5 * (p2[0, 0] + p2[0, 3]) + offset_x)
        x2 = int(0.5 * (p2[0, 1] + p2[0, 2]) + offset_x)
        y1 = int(0.5 * (p2[1, 0] + p2[1, 1]) + offset_y)
        y2 = int(0.5 * (p2[1, 2] + p2[1, 3]) + offset_y)

        x1 = int(min(width - 1, max(0, x1)))
        x2 = int(min(width - 1, max(x1, x2)))
        y1 = int(min(height - 1, max(0, y1)))
        y2 = int(min(height - 1, max(y1, y2)))

        cv2.rectangle(warp_rgb, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    cv2.imwrite('/home/junha/Downloads/{}_orig.jpg'.format(image_id), orig_rgb)
    cv2.imwrite('/home/junha/Downloads/{}_warp.jpg'.format(image_id), warp_rgb)

    # zz = convert_depth_float_from_uint8(cv2.imread(str(in_depth_path), cv2.IMREAD_UNCHANGED))
    # with open(str(in_intrinsic_path), 'r') as file:
    #     intrinsic = json.load(file)
    # xx = np.linspace(0, zz.shape[1] - 1, zz.shape[1], dtype=np.float32)
    # yy = np.linspace(0, zz.shape[0] - 1, zz.shape[0], dtype=np.float32)
    # xx, yy = np.meshgrid(xx, yy)
    # xx = (xx - intrinsic['cx']) * zz / intrinsic['fx']
    # yy = (yy - intrinsic['cy']) * zz / intrinsic['fy']
    # xyz = np.stack((xx, yy, zz), axis=-1)
    # mask = zz > 0

    # import open3d as o3d
    # obj_list = []
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pts)
    # obj_list.append(pcd)
    # for _, v in aabb_dict[image_id]:
    #     bbox = create_line_mesh_from_center_size(np.array([v[0], -v[1], -v[2], v[3], v[4], v[5]]))
    #     obj_list += bbox.cylinder_segments
    # o3d.visualization.draw_geometries(obj_list)


def update_meta_to_concise_directory():
    image_id_list = ['{:06d}'.format(i) for i in range(1, 10336)]
    aabb_dict = json.load(open('/home/junha/data/sunrefer/xyzrgb/aabb.json', 'r'))
    orig_meta_dict = torch.load('/home/junha/data/sunrefer/meta.pt')
    new_meta_dict = dict()

    for image_id in image_id_list:
        in_intrinsic_path = out_dir / '{}.json'.format(image_id)
        intrinsic = json.load(open(str(in_intrinsic_path), 'r'))
        aabb_list = aabb_dict[image_id]

        orig_entry = orig_meta_dict[image_id]

        new_entry = intrinsic
        new_entry['E'] = orig_entry['E']
        obj_2d_list = orig_entry['obj_2d']
        new_obj_2d_list = []
        for obj_item in obj_2d_list:
            x1, y1, w, h = fetch_warped_bbox(obj_item['bbox_2d'], intrinsic)
            obj_item['bbox_2d'] = [x1, y1, w, h]
            new_obj_2d_list.append(obj_item)
        new_entry['obj_2d'] = new_obj_2d_list

        obj_3d_list = orig_entry['obj_3d']
        new_obj_3d_list = []
        for i, obj_item in enumerate(obj_3d_list):
            cls_name, v = aabb_list[i]
            assert cls_name == obj_item['class_name']
            obj_item['aabb'] = [v[0], -v[1], -v[2], v[3], v[4], v[5]]
            new_obj_3d_list.append(obj_item)
        new_entry['obj_3d'] = new_obj_3d_list

    torch.save(new_meta_dict, '/home/junha/data/sunrefer/xyzrgb_concise/meta.pt')


def update_new_meta():
    new_meta_dict = torch.load('/home/junha/data/sunrefer/xyzrgb_concise/meta.pt')
    aabb_dict = json.load(open('/home/junha/data/sunrefer/xyzrgb/aabb.json', 'r'))
    for image_id in new_meta_dict.keys():
        obj_3d_list = new_meta_dict[image_id]['obj_3d']
        aabb_list = aabb_dict[image_id]
        assert len(obj_3d_list) == len(aabb_list)
        for i, (cls_name, v) in enumerate(aabb_list):
            assert cls_name == obj_3d_list[i]['class_name']
            obj_3d_list[i]['aabb'] = [v[0], -v[1], -v[2], v[3], v[4], v[5]]
        new_meta_dict[image_id]['obj_3d'] = obj_3d_list

    torch.save(new_meta_dict, '/home/junha/data/sunrefer/xyzrgb_concise/meta2.pt')


class Object2DInformation:
    def __init__(self, item):
        self.class_name: str = item['class_name']
        self.has_3d: bool = item['has_3d_bbox']
        self.bbox: List[float] = item['bbox_2d']


class Object3DInformation:
    def __init__(self, item):
        self.class_name: str = item['class_name']
        self.basis: np.array = item['basis']
        self.coeffs: np.array = item['coeffs']
        self.centroid: np.array = item['centroid']
        self.orientation: np.array = item['orientation']
        self.aabb: np.array = np.array(item['aabb'])


class MetaInformation:
    def __init__(self, item):
        self.width: int = item['width']
        self.height: int = item['height']
        self.fx: float = item['fx']
        self.fy: float = item['fy']
        self.cx: float = item['cx']
        self.cy: float = item['cy']
        self.offset_x: float = item['offset_x']
        self.offset_y: float = item['offset_y']
        self.H: np.array = np.reshape(np.array(item['H'], dtype=np.float32), (3, 3))
        self.E: np.array = item['E'].astype(dtype=np.float32)
        self.obj_2d_list = list(map(Object2DInformation, item['obj_2d']))
        self.obj_3d_list = list(map(Object3DInformation, item['obj_3d']))


def load_meta_information() -> Dict[str, MetaInformation]:
    meta_dict = torch.load('/home/junha/data/sunrefer/xyzrgb_concise/meta.pt')
    return {k: MetaInformation(v) for k, v in meta_dict.items()}


def update_sunrefer_concise():
    meta_dict = load_meta_information()
    orig_refer = json.load(open('/home/junha/data/sunrefer/SUNREFER_v2_revised.json', 'r'))
    new_refer_dict = dict()

    for anno_id in sorted(orig_refer.keys()):
        image_id, object_id = anno_id.split('_')
        object_id = int(object_id)
        refer_entry = orig_refer[anno_id]
        object_entry = meta_dict[image_id].obj_2d_list[int(object_id)]
        if refer_entry['object_name'] != object_entry.class_name:
            print(anno_id, refer_entry['object_name'], object_entry.class_name)
        refer_entry['bbox2d'] = object_entry.bbox
        new_refer_dict[anno_id] = refer_entry

    with open('/home/junha/data/sunrefer/xyzrgb_concise/sunrefer.json', 'w') as file:
        json.dump(new_refer_dict, file, indent=4)


def test_sunrefer_concise():
    refer_dict = json.load(open('/home/junha/data/sunrefer/xyzrgb_concise/sunrefer.json', 'r'))
    meta_dict: Dict[str, MetaInformation] = load_meta_information()

    for anno_id, refer_entry in refer_dict.items():
        print(anno_id)
        image_id, object_id = anno_id.split('_')
        object_id = int(object_id)
        bbox = refer_entry['bbox2d']

        meta_entry = meta_dict[image_id]
        target_object_entry = meta_entry.obj_3d_list[object_id]
        print('target bbox2d', bbox)
        print('target bbox3d aabb', target_object_entry.aabb)

        raw_depth = cv2.imread('/home/junha/data/sunrefer/xyzrgb_concise/{}.png'.format(image_id), cv2.IMREAD_UNCHANGED)
        depth = convert_depth_float_from_uint8(raw_depth)
        pcd, mask, rx, ry = unproject(depth, meta_entry.fx, meta_entry.fy, meta_entry.cx, meta_entry.cy)
        print(pcd.shape)

        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        x1 = max(0, min(meta_entry.width - 1, x1))
        x2 = max(x1, min(meta_entry.width, x2))
        y1 = max(0, min(meta_entry.height - 1, y1))
        y2 = max(y1, min(meta_entry.height, y2))
        pcd_crop = pcd[y1:y2, x1:x2, :]
        depth_crop = depth[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]
        pcd_sample = pcd_crop[mask_crop, :]
        depth_sample = depth_crop[mask_crop]
        print(pcd_sample.shape)

        print(np.min(pcd_sample[:, 2]), np.max(pcd_sample[:, 2]))
        print(np.min(depth_sample), np.max(depth_sample))

        # todo: create pointcloud dataset
        # todo: create image patch dataset

        break


if __name__ == '__main__':
    # image_id_list = ['{:06d}'.format(i) for i in range(1, 10336)]
    # pool = Pool(20)
    # result_list_tqdm = []
    # for result in tqdm(pool.imap(fetch_pixel_depth, image_id_list), total=len(image_id_list)):
    #     continue

    # update_meta_to_concise_directory()
    # update_new_meta()
    # meta_dict = read_meta_information()
    # for k, v in list(meta_dict.items())[:2]:
    #     for i, (o1, o2) in enumerate(zip(v.obj_2d_list, v.obj_3d_list)):
    #         print(k, i, o1.bbox, o2.aabb)
    # update_sunrefer_concise()
    test_sunrefer_concise()

    # visualize_with_gt_box('010333')
    # fetch_pixel_depth('000001')
    # visualize_pixel_depth('000001')

    # tm = TransformManager()
    # tm.visualize_pcds('000001')
    # tm.visualize_pcds('004131')
    # tm.visualize_pcds('003000')
    # tm.visualize_pcds('009566')
    # estimate_homography_from_extrinsic('000001')
