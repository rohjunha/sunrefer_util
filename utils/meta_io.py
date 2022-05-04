from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np
import scipy.io

from utils.directory import fetch_official_sunrgbd_dir, fetch_meta_v1_path, fetch_meta_2d_path, fetch_meta_3d_path


class BoundingBox3DV1:
    def __init__(self, info):
        self.basis, self.coeffs, self.centroid, _class_name, _, _, self.orientation, _, _ = info
        self.class_name = _class_name[0]

    def __repr__(self):
        return 'BoundingBox3DV1 - class_name: {}, basis: {}, coeffs: {}, centroid: {}, orientation: {}'.format(
            self.class_name, self.basis, self.coeffs, self.centroid, self.orientation)


class BoundingBox3DV2:
    def __init__(self, info):
        self.basis, self.coeffs, self.centroid, _class_name, _, self.orientation, label = info
        self.class_name = _class_name[0]

    def __repr__(self):
        return 'BoundingBox3DV2 - class_name: {}, basis: {}, coeffs: {}, centroid: {}, orientation: {}'.format(
            self.class_name, self.basis, self.coeffs, self.centroid, self.orientation)


class BoundingBox2DV2:
    def __init__(self, info):
        _object_id, _gt_bbox_2d, _class_name, _has_3d_bbox = info
        self.object_id = int(_object_id[0][0])
        self.class_name = str(_class_name[0])
        self.has_3d_bbox = bool(_has_3d_bbox[0][0])
        self.gt_bbox_2d = _gt_bbox_2d[0, :].astype(np.int64).tolist()

    def __repr__(self):
        return 'BoundingBox2DV2 - object_id: {}, class_name: {}, has_3d_bbox: {}, gt_bbox_2d: {}'.format(
            self.object_id, self.class_name, self.has_3d_bbox, self.gt_bbox_2d)


class MetaObjectBase:
    def __init__(self, seq_name: str, rgb_name: str, depth_name: str):
        self.seq_root = fetch_official_sunrgbd_dir() / seq_name
        self.rgb_name = rgb_name
        self.depth_name = depth_name
        assert self.rgb_path.exists()
        assert self.depth_path.exists()
        assert self.seg_path.exists()

    def __repr__(self):
        return 'seq_name: {}'.format(self.seq_root.stem)

    @property
    def depth_path(self) -> Path:
        if self.seq_root is not None and self.depth_name is not None:
            return self.seq_root / 'depth' / self.depth_name
        else:
            return Path()

    @property
    def rgb_path(self) -> Path:
        if self.seq_root is not None and self.rgb_name is not None:
            return self.seq_root / 'image' / self.rgb_name
        else:
            return Path()

    @property
    def seg_path(self) -> Path:
        if self.seq_root is not None:
            return self.seq_root / 'seg.mat'
        else:
            return Path()

    def create_all_binary_segmentations(self):
        seg_data = scipy.io.loadmat(str(self.seg_path))
        seg_label = seg_data['seglabel']

        data = []
        for i in range(seg_data['names'].shape[1]):
            name = seg_data['names'][0, i][0]
            seg_image = (seg_label == i + 1).astype(np.uint8) * 255
            data.append((i, name, seg_image))
        return data

    def create_conditioned_binary_segmentations(self, coord):
        seg_data = scipy.io.loadmat(str(self.seg_path))
        seg_label = seg_data['seglabel']
        x1, y1, w, h = coord

        data = []
        for i in range(seg_data['names'].shape[1]):
            name = seg_data['names'][0, i][0]
            seg_image = (seg_label == i + 1).astype(np.uint8)
            seg_sum = np.sum(seg_image[y1:y1 + h, x1:x1 + w])
            if seg_sum > 0:
                data.append((i, name, seg_image * 255, seg_sum / (w * h), seg_sum / np.sum(seg_image)))
        return data


def draw_bbox_and_text(image, bbox):
    x1, y1, w, h = bbox.gt_bbox_2d
    cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
    cv2.putText(image, bbox.class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


class MetaObject2DV2(MetaObjectBase):
    def __init__(self, item):
        seq_name, gt_2d_bbox, depth_path, rgb_path, depth_name, rgb_name, sensor_type = item
        MetaObjectBase.__init__(self, str(seq_name[0]), str(rgb_name[0]), str(depth_name[0]))
        self.gt_2d_bbox = [BoundingBox2DV2(gt_2d_bbox[0, i]) for i in range(gt_2d_bbox.shape[1])]

    def __repr__(self):
        return 'MetaObject2DV2 - {}, gt_2d_bbox: [{}]'.format(
            MetaObjectBase.__repr__(self), ', '.join([str(b) for b in self.gt_2d_bbox]))

    def draw_bbox(self, object_id: str):
        image = cv2.cvtColor(cv2.imread(str(self.rgb_path)), cv2.COLOR_BGR2RGB)
        bbox = self.gt_2d_bbox[int(object_id)]
        draw_bbox_and_text(image, bbox)
        return image

    def draw_bbox_all(self):
        image = cv2.cvtColor(cv2.imread(str(self.rgb_path)), cv2.COLOR_BGR2RGB)
        for bbox in self.gt_2d_bbox:
            draw_bbox_and_text(image, bbox)
        return image

    def build_class_dict(self) -> Dict[str, List[BoundingBox2DV2]]:
        class_dict = defaultdict(list)
        for i, bbox in enumerate(self.gt_2d_bbox):
            class_dict[bbox.class_name].append(bbox)
        return class_dict


class MetaObject3DV2(MetaObjectBase):
    def __init__(self, item):
        seq_name, Rt, K, depth_path, rgb_path, anno_extrinsics, depth_name, rgb_name, sensor_type, valid, gt_3d_bbox = item
        MetaObjectBase.__init__(self, str(seq_name[0]), str(rgb_name[0]), str(depth_name[0]))
        self.valid = bool(valid[0][0] == 1)
        self.gt_3d_bbox = [BoundingBox3DV2(gt_3d_bbox[0, i]) for i in range(gt_3d_bbox.shape[1])]
        self.Rt = Rt
        self.K = K
        self.sensor_type = sensor_type
        self.extrinsics = anno_extrinsics

    def __repr__(self):
        return 'MetaObject3DV2 - {}, valid: {}, sensor_type: {}, extrinsics: {}, gt_3d_bbox: [{}]'.format(
            MetaObjectBase.__repr__(self), self.valid, self.sensor_type, self.extrinsics,
            ', '.join([str(b) for b in self.gt_3d_bbox]))


class MetaObjectV1(MetaObjectBase):
    def __init__(self, item):
        seq_name, gt_3d_bbox, _, _, depth_path, rgb_path, anno_extrinsics, depth_name, rgb_name, sensor_type, valid, gt_corner_3d, gt_2d_bbox = item
        MetaObjectBase.__init__(self, str(seq_name[0]), str(rgb_name[0]), str(depth_name[0]))
        self.valid = bool(valid[0][0] == 1)
        self.gt_corner_3d = gt_corner_3d  # float64, (14, )
        self.gt_2d_bbox = [gt_2d_bbox[0, i][0] for i in range(gt_2d_bbox.shape[1])]
        self.gt_3d_bbox = [BoundingBox3DV1(gt_3d_bbox[0, i]) for i in range(gt_3d_bbox.shape[1])]

    def __repr__(self):
        return 'MetaObjectV1 - {}, valid: {}, gt_corner_3d: {}, gt_2d_bbox: [{}], gt_3d_bbox: [{}]'.format(
            MetaObjectBase.__repr__(self), self.valid, self.gt_corner_3d,
            ', '.join([str(b) for b in self.gt_2d_bbox]),
            ', '.join([str(b) for b in self.gt_3d_bbox]))

    def create_binary_segmentation(self, object_id: str):
        seg_data = scipy.io.loadmat(str(self.seg_path))
        seg_label = seg_data['seglabel']

        class_name = self.gt_3d_bbox[int(object_id)].class_name
        print(len(self.gt_3d_bbox), object_id, class_name)
        class_idx = -1
        for i in range(seg_data['names'].shape[1]):
            name = seg_data['names'][0, i]
            print(i, name)
            if name == class_name:
                class_idx = i + 1
                break
        assert class_idx >= 0
        return (seg_label == class_idx).astype(np.uint8) * 255


def fetch_scene_object_by_image_id(version: str):
    if version == 'v1':
        meta_path = fetch_meta_v1_path()
        meta_key = 'SUNRGBDMeta'
        object_cls = MetaObjectV1
    elif version == 'v2_2d':
        meta_path = fetch_meta_2d_path()
        meta_key = 'SUNRGBDMeta2DBB'
        object_cls = MetaObject2DV2
    elif version == 'v2_3d':
        meta_path = fetch_meta_3d_path()
        meta_key = 'SUNRGBDMeta'
        object_cls = MetaObject3DV2
    else:
        raise KeyError('invalid version: {}'.format(version))

    data = scipy.io.loadmat(str(meta_path))[meta_key]
    object_by_image_id = dict()
    for i in range(data.shape[1]):
        image_id = '{:06d}'.format(i + 1)
        obj = object_cls(data[0, i])
        object_by_image_id[image_id] = obj
    return object_by_image_id
