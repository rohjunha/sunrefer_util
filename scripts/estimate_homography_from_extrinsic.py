import json
from pathlib import Path
from shutil import copyfile

import cv2
import numpy as np
import torch

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


class TransformManager:
    def __init__(self):
        self.meta_dict = torch.load('/home/junha/data/sunrefer/meta.pt')
        self.intrinsic_fetcher = IntrinsicFetcher()
        self.scene_by_image_id = fetch_scene_object_by_image_id('v2_3d')
        self.aabb_dict = json.load(open('/home/junha/data/sunrefer/xyzrgb/aabb.json', 'r'))

    def fetch_raw_rgb(self, image_id: str):
        raw_bgr = cv2.imread('/home/junha/data/sunrefer/rgb/{}.jpg'.format(image_id), cv2.IMREAD_COLOR)
        return cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    def fetch_raw_depth(self, image_id: str):
        return cv2.imread('/home/junha/data/sunrefer/raw_depth/{}.png'.format(image_id), cv2.IMREAD_UNCHANGED)

    def fetch_original_pcd(self, image_id: str):
        fx, fy, cx, cy, h, w = self.intrinsic_fetcher[image_id]
        E = np.array(self.meta_dict[image_id]['E'], dtype=np.float32)
        raw_depth = self.fetch_raw_depth(image_id)
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

        raw_depth = self.fetch_raw_depth(image_id)
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

        raw_rgb = normalize_rgb(self.fetch_raw_rgb(image_id))
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
        print(intrinsic)

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

        offset_x = -np.min(p2[0, :])
        offset_y = -np.min(p2[1, :])
        p2[0, :] += offset_x
        p2[1, :] += offset_y
        width = min(2000, int(round(np.max(p2[0, :]))))
        height = min(2000, int(round(np.max(p2[1, :]))))
        cx += offset_x
        cy += offset_y

        raw_rgb = normalize_rgb(self.fetch_raw_rgb(image_id))
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
        mask = sampled_depth > 0
        rx = np.linspace(0., 1., sampled_depth.shape[1], dtype=np.float32)
        ry = np.linspace(0., 1., sampled_depth.shape[0], dtype=np.float32)
        rx, ry = np.meshgrid(rx, ry)

        depth_ = sampled_depth[..., np.newaxis]
        mask_ = mask[..., np.newaxis].astype(dtype=np.float32)
        rx_ = rx[..., np.newaxis]
        ry_ = ry[..., np.newaxis]
        data = np.concatenate((depth_, rx_, ry_, sampled_rgb, mask_), axis=-1)
        data[~mask, :] = 0.

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


def pcd_from_depth(raw_depth, E, fx, fy, cx, cy):
    if raw_depth.dtype == np.uint16:
        depth = np.bitwise_or(np.right_shift(raw_depth, 3), np.left_shift(raw_depth, 13)).astype(np.float32) / 1000.
        depth[depth > 7.0] = 0.
    else:
        depth = raw_depth
    mask = depth > 0
    mx = np.linspace(0, depth.shape[1] - 1, depth.shape[1], dtype=np.float32)
    my = np.linspace(0, depth.shape[0] - 1, depth.shape[0], dtype=np.float32)
    rx = np.linspace(0., 1., depth.shape[1], dtype=np.float32)
    ry = np.linspace(0., 1., depth.shape[0], dtype=np.float32)
    rx, ry = np.meshgrid(rx, ry)
    xx, yy = np.meshgrid(mx, my)
    xx = (xx - cx) * depth / fx
    yy = (yy - cy) * depth / fy
    pcd = np.stack((xx, yy, depth), axis=-1)
    if E is not None:
        pcd = pcd @ np.transpose(E)
    return pcd, mask, rx, ry


def estimate_homography_from_extrinsic(image_id: str):
    fx, fy, cx, cy, h, w = intrinsic_fetcher[image_id]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    E = np.array(meta_dict[image_id]['E'], dtype=np.float32)
    T = (K @ E @ np.linalg.inv(K)).astype(dtype=np.float32)
    T = T / T[2, 2]

    p1 = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]], dtype=np.float32)
    p2 = T @ p1
    p2 = np.divide(p2[:2, :], np.tile(p2[-1, :], (2, 1)))

    offset_x = -np.min(p2[0, :])
    offset_y = -np.min(p2[1, :])

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
    print(intrinsic)

    raw_rgb = normalize_rgb(cv2.imread('/home/junha/data/sunrefer/rgb/{}.jpg'.format(image_id)))

    raw_depth_ = cv2.imread('/home/junha/data/sunrefer/raw_depth/{}.png'.format(image_id), cv2.IMREAD_UNCHANGED)
    raw_depth_ = np.bitwise_or(np.right_shift(raw_depth_, 3), np.left_shift(raw_depth_, 13)).astype(np.float32) / 1000.
    raw_depth_[raw_depth_ > 7.0] = 0.
    raw_mask_ = raw_depth_ > 0

    raw_mask = raw_mask_.astype(np.float32)[..., np.newaxis]
    raw_depth = raw_depth_[..., np.newaxis]

    raw_data = np.concatenate((raw_depth, raw_rgb, raw_mask), axis=-1)
    data = cv2.warpPerspective(raw_data, T, (width, height))

    # pcd, mask, rx, ry = pcd_from_depth(data[..., 0], None, fx, fy, cx, cy)

    mask = data[..., -1] > 0.95
    depth = data[..., 0]
    mx = np.linspace(0, depth.shape[1] - 1, depth.shape[1], dtype=np.float32)
    my = np.linspace(0, depth.shape[0] - 1, depth.shape[0], dtype=np.float32)
    rx = np.linspace(0., 1., depth.shape[1], dtype=np.float32)
    ry = np.linspace(0., 1., depth.shape[0], dtype=np.float32)
    rx, ry = np.meshgrid(rx, ry)
    xx, yy = np.meshgrid(mx, my)
    xx = (xx - cx) * depth / fx
    yy = (yy - cy) * depth / fy
    pcd = np.stack((xx, yy, depth), axis=-1)

    rgb = denormalize_rgb(data[..., 1:4])
    cv2.imwrite('/home/junha/Downloads/{}_h.jpg'.format(image_id), rgb)



    sampled_mask = np.zeros((height, width), dtype=bool)
    sampled_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    sampled_depth = np.zeros((height, width), dtype=np.float32)

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

    sampled_rgb[yy, xx, :] = rgb[grid_y, grid_x, :]
    sampled_mask[yy, xx] = True
    sampled_depth[yy, xx] = pcd[grid_y, grid_x, 2]

    pcd_pixel, mask_pixel, _, _ = pcd_from_depth(sampled_depth, None, fx, fy, cx, cy)
    print(pcd_pixel.shape, mask_pixel.shape)





    import open3d as o3d
    pt1 = o3d.geometry.PointCloud()
    pt2 = o3d.geometry.PointCloud()
    pt3 = o3d.geometry.PointCloud()
    pt1.points = o3d.utility.Vector3dVector(pcd_orig[mask_orig, :])
    pt2.points = o3d.utility.Vector3dVector(pcd[mask, :])
    pt3.points = o3d.utility.Vector3dVector(pcd_pixel[mask_pixel, :])
    pt1.paint_uniform_color((1, 0, 0))
    pt2.paint_uniform_color((0, 1, 0))
    pt3.paint_uniform_color((0, 0, 1))
    # pt1.estimate_normals()
    # pt2.estimate_normals()
    #
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     pt2, pt1, 0.02, np.eye(4, dtype=np.float32),
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)

    obj_list = [pt1, pt2, pt3]
    for i, (_, aabb) in enumerate(aabb_dict[image_id]):
        cx_, cy_, cz_, sx, sy, sz = aabb
        line_mesh = create_line_mesh_from_center_size(np.array([cx_, -cy_, -cz_, sx, sy, sz]))
        obj_list += line_mesh.cylinder_segments

    o3d.visualization.draw_geometries(obj_list)
    # image = np.zeros((height, width, 3), dtype=np.uint8)
    # draw_registration_result(pt2, pt1, reg_p2p.transformation)


if __name__ == '__main__':
    tm = TransformManager()
    tm.visualize_pcds('000001')
    # tm.visualize_pcds('009566')
    # estimate_homography_from_extrinsic('000001')
