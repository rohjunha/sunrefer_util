from pathlib import Path
from typing import Tuple

import numpy as np

from utils.geometry import convert_depth_float_from_uint8, unproject, transform
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


class PointCloudDataset:
    def __init__(
            self,
            db_path: Path = Path.home() / 'data/sunrefer/xyzrgb_concise/pcd'):
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

    def fetch_pcd(self, image_id: str):
        scene: MetaInformation = self.scene_dict[image_id]
        depth = convert_depth_float_from_uint8(self.storage.get_depth(image_id))
        pcd, mask, rx, ry = unproject(float_depth=depth, fx=scene.fx, fy=scene.fy, cx=scene.cx, cy=scene.cy)
        pcd = transform(pcd, scene.E)
        return pcd, mask, rx, ry

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
        x1, x2, y1, y2 = convert_bbox_into_slices(bbox, *pcd.shape[:2])
        return xyzrgb[y1:y2, x1:x2, :], mask[y1:y2, x1:x2]

    def fetch_sampled_points(self, image_id: str, bbox, num_samples: int):
        xyz, mask = self.fetch_cropped_xyzrgb(image_id, bbox)
        vec_xyz = xyz[mask, :]
        return vec_xyz[np.random.choice(vec_xyz.shape[0], num_samples), :]


if __name__ == '__main__':
    dataset = PointCloudDataset()
    image_id = '000001'
    bbox = [100, 100, 50, 100]
    pcd, _, _, _ = dataset.fetch_pcd(image_id)
    print(pcd.shape)
    pcd_crop, _, _, _ = dataset.fetch_cropped_pcd(image_id, bbox)
    xyzrgb, _ = dataset.fetch_cropped_xyzrgb(image_id, bbox)
    print(xyzrgb.shape)
    samples = dataset.fetch_sampled_points(image_id, bbox, 100)
    print(samples.shape)
