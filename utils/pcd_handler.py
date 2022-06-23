import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from utils.directory import fetch_concise_storage_path, fetch_concise_correspondence_path
from utils.geometry import convert_depth_float_from_uint8, unproject
from utils.scene import load_scene_information, MetaInformation
from utils.storage import PointCloudStorage


def convert_bbox_into_slices(bbox, height, width):
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    x1 = int(max(0, min(width - 1, x1)))
    x2 = int(max(x1, min(width, x2)))
    y1 = int(max(0, min(height - 1, y1)))
    y2 = int(max(y1, min(height, y2)))
    return x1, x2, y1, y2


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


class PointCloudHandler:
    def __init__(
            self,
            db_path: Path = None):
        if db_path is None:
            db_path = fetch_concise_storage_path()
        assert db_path.exists()
        self.storage = PointCloudStorage(read_only=True, db_path=db_path)
        self.scene_dict = load_scene_information()

    def fetch_raw_rgb(self, image_id: str):
        return self.storage.get_rgb(image_id)

    def fetch_cropped_raw_rgb(self, image_id: str, bbox):
        rgb = self.fetch_raw_rgb(image_id)
        x1, x2, y1, y2 = convert_bbox_into_slices(bbox, *rgb.shape[:2])
        return rgb[y1:y2, x1:x2, :]

    def fetch_normalized_rgb(self, image_id: str):
        return normalize_rgb(self.fetch_raw_rgb(image_id))

    def fetch_cropped_normalized_rgb(self, image_id: str, bbox):
        rgb = self.fetch_normalized_rgb(image_id)
        x1, x2, y1, y2 = convert_bbox_into_slices(bbox, *rgb.shape[:2])
        return rgb[y1:y2, x1:x2, :]

    def fetch_raw_depth(self, image_id: str):
        return self.storage.get_depth(image_id)

    def fetch_pcd(self, image_id: str) -> Tuple[np.array, np.array, np.array, np.array]:
        scene: MetaInformation = self.scene_dict[image_id]
        depth = convert_depth_float_from_uint8(self.storage.get_depth(image_id))
        return unproject(float_depth=depth, fx=scene.fx, fy=scene.fy, cx=scene.cx, cy=scene.cy)

    def fetch_cropped_pcd(self, image_id: str, bbox):
        pcd, mask, rx, ry = self.fetch_pcd(image_id)
        x1, x2, y1, y2 = convert_bbox_into_slices(bbox, *pcd.shape[:2])
        pcd_crop = pcd[y1:y2, x1:x2, :]
        mask_crop = mask[y1:y2, x1:x2]
        rx_crop = rx[y1:y2, x1:x2]
        ry_crop = ry[y1:y2, x1:x2]
        return pcd_crop, mask_crop, rx_crop, ry_crop

    def fetch_xyzrgb(self, image_id: str) -> Tuple[np.array, np.array]:
        pcd, mask, rx, ry = self.fetch_pcd(image_id)
        rgb = self.fetch_normalized_rgb(image_id)
        xyzrgb = np.concatenate((pcd, rx[..., np.newaxis], ry[..., np.newaxis], rgb), axis=-1)  # (h, w, 8)
        xyzrgb[~mask, :] = 0.
        return xyzrgb, mask

    def fetch_cropped_xyzrgb(self, image_id: str, bbox) -> Tuple[np.array, np.array]:
        xyzrgb, mask = self.fetch_xyzrgb(image_id)
        x1, x2, y1, y2 = convert_bbox_into_slices(bbox, *xyzrgb.shape[:2])
        return xyzrgb[y1:y2, x1:x2, :], mask[y1:y2, x1:x2]

    def fetch_sampled_points(self, image_id: str, bbox, num_samples: int):
        xyz, mask = self.fetch_cropped_xyzrgb(image_id, bbox)
        vec_xyz = xyz[mask, :]
        return vec_xyz[np.random.choice(vec_xyz.shape[0], num_samples), :]


class PointCloudBoundingBoxHandler(PointCloudHandler):
    def __init__(
            self,
            split: str,
            db_path: Path = None):
        PointCloudHandler.__init__(self, db_path)
        assert split in {'train', 'test', 'val'}
        self.num_points = 1000
        self.split = split
        image_id_range = range(5051, 10336) if split == 'train' else range(1, 5051)
        self.image_id_list = ['{:06d}'.format(i) for i in image_id_range]
        with open(str(fetch_concise_correspondence_path()), 'r') as file:
            self.corr_dict = json.load(file)
        self.item_list = []
        for image_id in self.image_id_list:
            self.item_list += [(image_id, i, j, cls_name) for i, j, cls_name in self.corr_dict[image_id]]

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, index):
        image_id, i, j, cls_name = self.item_list[index]
        scene = self.scene_dict[image_id]
        obj_2d = scene.obj_2d_list[i]
        sampled_points = self.fetch_sampled_points(image_id, obj_2d.bbox, self.num_points)  # (#points, 8)
        obj_3d = scene.obj_3d_list[j]
        return {
            'points': sampled_points,
            'bbox': obj_2d.bbox,
            'aabb': obj_3d.aabb,
            'fx': scene.fx,
            'fy': scene.fy,
            'cx': scene.cx,
            'cy': scene.cy,
            'width': scene.width,
            'height': scene.height,
        }


if __name__ == '__main__':
    dataset = PointCloudHandler()
    image_id = '010000'

    image = dataset.fetch_raw_rgb(image_id)
    for obj_2d in dataset.scene_dict[image_id].obj_2d_list:
        x1, y1, w, h = obj_2d.bbox
        cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), color=(0, 0, 255), thickness=2)
    cv2.imwrite('/home/junha/Downloads/{}_verify.jpg'.format(image_id), image)

    # bbox = [100, 100, 50, 100]
    # pcd, _, _, _ = dataset.fetch_pcd(image_id)
    # print(pcd.shape)
    # pcd_crop, _, _, _ = dataset.fetch_cropped_pcd(image_id, bbox)
    # xyzrgb, _ = dataset.fetch_cropped_xyzrgb(image_id, bbox)
    # print(xyzrgb.shape)
    # samples = dataset.fetch_sampled_points(image_id, bbox, 100)
    # print(samples.shape)
