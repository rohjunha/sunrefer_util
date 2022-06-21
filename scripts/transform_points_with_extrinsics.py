import json
from math import isnan
from multiprocessing import Pool
from pathlib import Path
from typing import Union, Any, Dict, List, Tuple

import numpy as np
import open3d as o3d
import torch
from scipy.interpolate import griddata
from tqdm import tqdm

from utils.directory import fetch_intrinsic_from_tag_path, fetch_segformer_path
from utils.intrinsic_fetcher import IntrinsicFetcher
from utils.line_mesh import LineMesh
from utils.meta_io import MetaObjectBase


def create_line_mesh_from_center_size(
        center_size: np.ndarray,
        color: Union[List[int], Tuple[int, int, int]] = (1, 0, 0),
        radius: float = 0.015):
    LINES = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
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


with open(str(fetch_intrinsic_from_tag_path()), 'r') as file:
    INTRINSIC_FROM_TAG = json.load(file)


class Object2D:
    def __init__(self, item: Dict[str, Any]):
        self.class_name = item['class_name']
        self.has_3d = item['has_3d_bbox']
        self.bbox = item['bbox_2d']


class Object3D:
    def __init__(self, item: Dict[str, Any]):
        self.class_name = item['class_name']
        self.basis = item['basis']
        self.coeffs = item['coeffs']
        self.centroid = item['centroid']
        self.orientation = item['orientation']


def convert_raw_depth(raw_depth):
    """Numpy implementation of SUNRGBD depth processing."""
    shift_depth = np.bitwise_or(np.right_shift(raw_depth, 3), np.left_shift(raw_depth, 13))
    float_depth = shift_depth.astype(dtype=np.float32) / 1000.
    float_depth[float_depth > 7.0] = 0.
    return float_depth


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


class SceneInformation(MetaObjectBase):
    def __init__(self, item: Dict[str, Any]):
        keys = ['seq_name', 'depth_name', 'rgb_name']
        MetaObjectBase.__init__(self, **{key: item[key] for key in keys})
        self.extrinsics = item['E']
        self.sensor_type = item['intrinsic']
        self.fx, self.fy, self.cx, self.cy, self.h, self.w = INTRINSIC_FROM_TAG[self.sensor_type]
        self.obj_3d_list = [Object3D(i) for i in item['obj_3d']]
        self.obj_2d_list = [Object2D(i) for i in item['obj_2d']]

    def project_2d(self, p):
        return p[0] / p[2] * self.fx + self.cx, p[1] / p[2] * self.fy + self.cy

    def load_pcd(self, apply_extrinsics: bool = True):
        raw_depth = self.load_depth()
        m = raw_depth > 0
        zz = convert_raw_depth(raw_depth)
        mx = np.linspace(0, raw_depth.shape[1] - 1, raw_depth.shape[1])
        my = np.linspace(0, raw_depth.shape[0] - 1, raw_depth.shape[0])
        rx = np.linspace(0., 1., raw_depth.shape[1])
        ry = np.linspace(0., 1., raw_depth.shape[0])
        rx, ry = np.meshgrid(rx, ry)
        xx, yy = np.meshgrid(mx, my)
        xx = (xx - self.cx) * zz / self.fx
        yy = (yy - self.cy) * zz / self.fy
        pcd = np.stack((xx, yy, zz), axis=-1)
        if apply_extrinsics:
            pcd = pcd @ np.transpose(self.extrinsics)
        return pcd, m, rx, ry


def pcd_from_depth(depth: np.array, fx, fy, cx, cy):
    zz = depth
    m = depth > 1e-5
    mx = np.linspace(0, depth.shape[1] - 1, depth.shape[1])
    my = np.linspace(0, depth.shape[0] - 1, depth.shape[0])
    rx = np.linspace(0., 1., depth.shape[1])
    ry = np.linspace(0., 1., depth.shape[0])
    rx, ry = np.meshgrid(rx, ry)
    xx, yy = np.meshgrid(mx, my)
    xx = (xx - cx) * zz / fx
    yy = (yy - cy) * zz / fy
    pcd = np.stack((xx, yy, zz), axis=-1)
    return pcd, m, rx, ry


SCENE_BY_IMAGE_ID = {k: SceneInformation(v) for k, v in torch.load(str(Path.home() / 'data/sunrefer/meta.pt')).items()}
OUT_DIR = Path.home() / 'data/sunrefer/xyzrgb_extrinsic'


def create_pcd_with_extrinsic(image_id: str):
    pcd_path = OUT_DIR / '{}.npy'.format(image_id)
    rgb_path = OUT_DIR / '{}.jpg'.format(image_id)
    intrinsic_path = OUT_DIR / '{}.json'.format(image_id)

    # if pcd_path.exists():
    #     return

    scene: SceneInformation = SCENE_BY_IMAGE_ID[image_id]
    rgb = scene.load_rgb()
    pcd, mask, rx, ry = scene.load_pcd(apply_extrinsics=True)

    seg_out = np.load(str(fetch_segformer_path(image_id)))
    seg_mask = np.ones(seg_out.shape, dtype=bool)
    for label in [0, 3, 5]:
        seg_mask[seg_out == label] = False

    mask_nz = np.logical_and(np.abs(pcd[..., 2]) > 1e-6, seg_mask)
    if np.sum(mask_nz) < 10000:
        mask_nz = np.abs(pcd[..., 2]) > 1e-6
    px = scene.fx * pcd[mask_nz, 0] / pcd[mask_nz, 2] + scene.cx
    py = scene.fy * pcd[mask_nz, 1] / pcd[mask_nz, 2] + scene.cy
    sampled_x = np.reshape(px, (-1,))
    sampled_y = np.reshape(py, (-1,))

    offset_x = round(-np.min(sampled_x)) if np.min(sampled_x) < 0 else 0
    offset_y = round(-np.min(sampled_y)) if np.min(sampled_y) < 0 else 0
    size_x = min(2000, round(np.max(sampled_x)))
    size_y = min(2000, round(np.max(sampled_y)))

    width = size_x + offset_x
    height = size_y + offset_y
    cx = scene.cx + (-np.min(sampled_x) if np.min(sampled_x) < 0 else 0.)
    cy = scene.cy + (-np.min(sampled_y) if np.min(sampled_y) < 0 else 0.)

    if width > 1000 or height > 1000:
        width = pcd.shape[1] + offset_x
        height = pcd.shape[0] + offset_y
        # width, height = pcd.shape[1], pcd.shape[0]
        # cx, cy = scene.cx, scene.cy

    intrinsics = {'width': width, 'height': height, 'cx': cx, 'cy': cy, 'fx': scene.fx, 'fy': scene.fy}
    print(image_id, intrinsics)

    sampled_mask = np.zeros((height, width), dtype=bool)
    sampled_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    sampled_depth = np.zeros((height, width), dtype=np.float32)

    xx_ = scene.fx * pcd[..., 0] / pcd[..., 2] + cx
    yy_ = scene.fy * pcd[..., 1] / pcd[..., 2] + cy
    mm = np.logical_and(~np.isnan(xx_), ~np.isnan(yy_))
    xx = np.round(xx_).astype(dtype=np.int32)
    yy = np.round(yy_).astype(dtype=np.int32)
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, pcd.shape[1] - 1, pcd.shape[1]),
        np.linspace(0, pcd.shape[0] - 1, pcd.shape[0]))
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

    pcd, mask, rx, ry = pcd_from_depth(sampled_depth, scene.fx, scene.fy, cx, cy)
    xyzrgb = np.concatenate((pcd, rx[..., np.newaxis], ry[..., np.newaxis], normalize_rgb(sampled_rgb)), axis=-1)
    xyzrgb[~mask, :] = 0.

    np.save(str(pcd_path), xyzrgb.astype(dtype=np.float32))
    o3d.io.write_image(str(rgb_path), o3d.geometry.Image(sampled_rgb))
    with open(str(intrinsic_path), 'w') as file:
        json.dump(intrinsics, file, indent=4)


class SunReferDataset:
    def __init__(self,
                 out_dir: Path = Path.home() / 'data/sunrefer/xyzrgb_extrinsic',
                 interp: bool = True,
                 meta_path: Path = Path.home() / 'data/sunrefer/meta.pt'):
        self.meta_path = meta_path
        self.scene_by_image_id = {k: SceneInformation(v) for k, v in torch.load(str(meta_path)).items()}
        self.intrinsic_fetcher = IntrinsicFetcher()
        self.interp = interp
        self.out_dir = out_dir

    def load_segformer_mask(self, image_id: str):
        seg_out = np.load(str(fetch_segformer_path(image_id)))
        seg_mask = np.ones(seg_out.shape, dtype=bool)
        for label in [0, 3, 5]:
            seg_mask[seg_out == label] = False
        return seg_mask

    def render_rgb_from_points(self, image_id: str):
        scene: SceneInformation = self.scene_by_image_id[image_id]
        rgb = scene.load_rgb()
        pcd, mask, rx, ry = scene.load_pcd(apply_extrinsics=True)
        # print('pcd loaded')
        mask_nz = np.logical_and(np.abs(pcd[..., 2]) > 1e-6, self.load_segformer_mask(image_id))
        px = scene.fx * pcd[mask_nz, 0] / pcd[mask_nz, 2] + scene.cx
        py = scene.fy * pcd[mask_nz, 1] / pcd[mask_nz, 2] + scene.cy
        sampled_x = np.reshape(px, (-1,))
        sampled_y = np.reshape(py, (-1,))
        sampled_r = np.reshape(rgb[mask_nz, 0], (-1,))
        sampled_g = np.reshape(rgb[mask_nz, 1], (-1,))
        sampled_b = np.reshape(rgb[mask_nz, 2], (-1,))
        sampled_d = np.reshape(pcd[mask_nz, 2], (-1,))

        offset_x = round(-np.min(sampled_x)) if np.min(sampled_x) < 0 else 0
        offset_y = round(-np.min(sampled_y)) if np.min(sampled_y) < 0 else 0
        size_x = round(np.max(sampled_x))
        size_y = round(np.max(sampled_y))

        width = size_x + offset_x
        height = size_y + offset_y
        cx = scene.cx + (-np.min(sampled_x) if np.min(sampled_x) < 0 else 0.)
        cy = scene.cy + (-np.min(sampled_y) if np.min(sampled_y) < 0 else 0.)

        # if width > 1000 or height > 1000:
        #     width = pcd.shape[1]
        #     height = pcd.shape[0]
        #     size_x = width
        #     size_y = height
        #     offset_x = 0
        #     offset_y = 0
        #     cx = scene.cx
        #     cy = scene.cy

        intrinsics = {'width': width, 'height': height, 'cx': cx, 'cy': cy, 'fx': scene.fx, 'fy': scene.fy}

        if self.interp:
            grid_x, grid_y = np.meshgrid(
                np.linspace(-offset_x, size_x - 1, width),
                np.linspace(-offset_y, size_y - 1, height))
            grid_r = griddata((sampled_x, sampled_y), sampled_r, (grid_x, grid_y))
            grid_g = griddata((sampled_x, sampled_y), sampled_g, (grid_x, grid_y))
            grid_b = griddata((sampled_x, sampled_y), sampled_b, (grid_x, grid_y))
            grid_d = griddata((sampled_x, sampled_y), sampled_d, (grid_x, grid_y))
            grid_rgb = np.stack((grid_r, grid_g, grid_b), axis=-1)

            sampled_rgb = np.zeros(grid_rgb.shape, dtype=np.uint8)
            sampled_depth = np.zeros(grid_d.shape, dtype=np.float32)
            nan_mask_rgb = ~np.isnan(grid_rgb)
            sampled_mask = ~np.isnan(grid_d)
            sampled_rgb[nan_mask_rgb] = np.round(grid_rgb[nan_mask_rgb])
            sampled_depth[sampled_mask] = grid_d[sampled_mask]
            # print('sampling is done.')

        # o3d.io.write_image('mask.png', o3d.geometry.Image(mask.astype(np.uint8) * 255))
        # o3d.io.write_image('sampled.png', o3d.geometry.Image(sampled_rgb))
        # np.save('depth.npy', sampled_depth)

        else:
            sampled_mask = np.zeros((height, width), dtype=bool)
            sampled_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            sampled_depth = np.zeros((height, width), dtype=np.float32)

            xx_ = scene.fx * pcd[..., 0] / pcd[..., 2] + cx
            yy_ = scene.fy * pcd[..., 1] / pcd[..., 2] + cy
            mm = np.logical_and(~np.isnan(xx_), ~np.isnan(yy_))
            xx = np.round(xx_).astype(dtype=np.int32)
            yy = np.round(yy_).astype(dtype=np.int32)
            grid_x, grid_y = np.meshgrid(
                np.linspace(0, pcd.shape[1] - 1, pcd.shape[1]),
                np.linspace(0, pcd.shape[0] - 1, pcd.shape[0]))
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

            # for rr in range(pcd.shape[0]):
            #     for cc in range(pcd.shape[1]):
            #         zz_ = pcd[rr, cc, 2]
            #         if abs(zz_) < 1e-5:
            #             continue
            #         xx_ = scene.fx * pcd[rr, cc, 0] / zz_ + (scene.cx + offset_x)
            #         yy_ = scene.fy * pcd[rr, cc, 1] / zz_ + (scene.cy + offset_y)
            #         if isnan(xx_) or isnan(yy_):
            #             continue
            #
            #         xx = int(xx_)
            #         yy = int(yy_)
            #         if 0 <= xx < sampled_rgb.shape[1] and 0 <= yy < sampled_rgb.shape[0]:
            #             sampled_rgb[yy, xx, :] = rgb[rr, cc, :]
            #             sampled_mask[yy, xx] = True
            #             sampled_depth[yy, xx] = zz_

        pcd, mask, rx, ry = pcd_from_depth(sampled_depth, scene.fx, scene.fy, cx, cy)
        xyzrgb = np.concatenate((pcd, rx[..., np.newaxis], ry[..., np.newaxis], normalize_rgb(sampled_rgb)), axis=-1)
        xyzrgb[~mask, :] = 0.

        pcd_path = self.out_dir / '{}.npy'.format(image_id)
        rgb_path = self.out_dir / '{}.jpg'.format(image_id)
        intrinsic_path = self.out_dir / '{}.json'.format(image_id)
        np.save(str(pcd_path), xyzrgb.astype(dtype=np.float32))
        o3d.io.write_image(str(rgb_path), o3d.geometry.Image(sampled_rgb))
        with open(str(intrinsic_path), 'w') as file:
            json.dump(intrinsics, file, indent=4)

    def compare_points(self, uniq_id: str, offset_x: float = 0., offset_y: float = 0.):
        image_id, object_id, anno_id = uniq_id.split('_')
        object_id = int(object_id)
        scene: SceneInformation = self.scene_by_image_id[image_id]

        pcd1_, mask1, rx, ry = scene.load_pcd(apply_extrinsics=True)
        print(pcd1_.shape, mask1.shape)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcd1_[mask1, :])
        o3d.io.write_point_cloud('pcd1.ply', pcd1)

        depth = np.load('depth.npy')
        pcd2_, mask2, _, _ = pcd_from_depth(depth, scene.fx, scene.fy, scene.cx, scene.cy, offset_x, offset_y)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pcd2_[mask2, :])
        o3d.io.write_point_cloud('pcd2.ply', pcd2)

    def visualize_pcd(self, image_id: str):
        pcd_path = self.out_dir / '{}.npy'.format(image_id)
        pcd = np.load(str(pcd_path))
        rgb = denormalize_rgb(pcd[..., -3:])
        o3d.io.write_image('vis{}.png'.format(image_id), o3d.geometry.Image(rgb))


def visualize_ply_files():
    pcd1 = o3d.io.read_point_cloud('pcd1.ply')
    pcd2 = o3d.io.read_point_cloud('pcd2.ply')
    pcd1.paint_uniform_color([1., 0., 0.])
    pcd2.paint_uniform_color([0., 1., 0.])
    # color1 = np.zeros((len(pcd1.points), 3), dtype=np.uint8)
    # color2 = np.zeros((len(pcd2.points), 3), dtype=np.uint8)
    # color1[:, 0] = 255
    # color2[:, 0] = 255
    # pcd1.color = color1
    # pcd2.color = color2
    geometries = [pcd1, pcd2]
    o3d.visualization.draw_geometries(geometries)

    # viewer = o3d.visualization.Visualizer()
    # viewer.create_window('viewer', visible=True, width=1920, height=1080)
    # viewer.add_geometry(pcd1)
    # cam = viewer.get_view_control().convert_to_pinhole_camera_parameters()
    # viewer.run()
    # viewer.capture_screen_image('screen1.png')
    # viewer.destroy_window()


if __name__ == '__main__':
    # visualize_ply_files()
    # dataset = SunReferDataset(interp=False)

    # dataset.render_rgb_from_points('001213')
    # dataset.visualize_pcd('001213')

    # image_id_list = ['{:06d}'.format(i) for i in range(1, 10336)]
    image_id_list = ['002803', '003654', '003701', '004131', '004281', '004767', '009566', '009587', '010079', '010199']
    pool = Pool(20)
    result_list_tqdm = []
    for result in tqdm(pool.imap(create_pcd_with_extrinsic, image_id_list), total=len(image_id_list)):
        pass

    # iou_by_uniq_id = {k: v for k, v in result_list_tqdm}
    #
    # filename = 'aabb={},seg_out={},bbox_ratio={},pcd_th={},mask_ratio={}.json'.format(
    #     aabb, use_seg_out_mask, bbox_2d_ratio, pcd_th_ratio, mask_ratio)
    # with open('/home/junha/data/sunrefer/iou/{}'.format(filename), 'w') as file:
    #     json.dump(iou_by_uniq_id, file, indent=4)
    #
    # t1 = time.time()
    # pcd_ = dataset.render_rgb_from_points('{:06d}'.format(i))
    # t2 = time.time()
    # print(t2 - t1)

    # with open('/home/junha/data/sunrefer/xyzrgb/aabb.json', 'r') as file:
    #     aabb = json.load(file)
    #
    # pcd = o3d.geometry.PointCloud()
    # mask = np.abs(pcd_[..., 0]) > 1e-5
    # # pcd_[..., 2] *= -1.
    # # Tyz = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    # # pcd_ = pcd_ @ np.transpose(Tyz)
    # pcd.points = o3d.utility.Vector3dVector(pcd_[..., :3][mask, :])
    #
    # o3d_obj_list = []
    # o3d_obj_list.append(pcd)
    # for cls, center_size in aabb['000001']:
    #     cx, cy, cz, sx, sy, sz = center_size
    #     center_size = cx, -cy, -cz, sx, sy, sz
    #     line_mesh = create_line_mesh_from_center_size(np.array(center_size))
    #     o3d_obj_list += line_mesh.cylinder_segments
    #
    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    # o3d_obj_list.append(coord_frame)
    # o3d.visualization.draw_geometries(o3d_obj_list)

    # dataset.compare_points('000001_1_0', 21.462692953200985, 0.)
