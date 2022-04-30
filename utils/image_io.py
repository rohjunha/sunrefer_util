import cv2
import numpy as np
import scipy.io
from utils.intrinsic_fetcher import IntrinsicFetcher

from utils.directory import fetch_depth_data_path, fetch_rgb_path


def fetch_depth_data(image_id: str) -> np.ndarray:
    return scipy.io.loadmat(fetch_depth_data_path(image_id))['instance']


def fetch_rgb(image_id: str) -> np.ndarray:
    return cv2.imread(str(fetch_rgb_path(image_id)), cv2.IMREAD_COLOR)


class PointcloudConverter:
    def __init__(self):
        self.im = IntrinsicFetcher()

    def __getitem__(self, image_id: str):
        fx, fy, x0, y0, h, w = self.im[image_id]
        depth = fetch_depth_data(image_id)
        valid_mask = depth > 0
        assert depth.shape[0] == h and depth.shape[1] == w
        cc, rr = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
        nx = (cc - x0) * depth / fx
        ny = (rr - y0) * depth / fy
        return np.stack((nx, ny, depth), axis=-1), valid_mask
