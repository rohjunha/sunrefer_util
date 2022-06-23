from typing import List, Dict

import numpy as np
import torch


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


def load_scene_information() -> Dict[str, MetaInformation]:
    meta_dict = torch.load('/home/junha/data/sunrefer/xyzrgb_concise/meta.pt')
    return {k: MetaInformation(v) for k, v in meta_dict.items()}
