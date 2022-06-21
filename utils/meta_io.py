from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import scipy.io

from utils.directory import fetch_official_sunrgbd_dir, fetch_meta_v1_path, fetch_meta_2d_path, fetch_meta_3d_path

CL = [[-1, -1, -1],
      [1, -1, -1],
      [-1, 1, -1],
      [1, 1, -1],
      [-1, -1, 1],
      [1, -1, 1],
      [-1, 1, 1],
      [1, 1, 1]]


class BoundingBox3DV1:
    def __init__(self, info):
        self.basis, self.coeffs, self.centroid, _class_name, _, _, self.orientation, _, _ = info
        self.class_name = _class_name[0]

    def __repr__(self):
        return 'BoundingBox3DV1 - class_name: {}, basis: {}, coeffs: {}, centroid: {}, orientation: {}'.format(
            self.class_name, self.basis, self.coeffs, self.centroid, self.orientation)


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


class BoundingBox3DV2:
    def __init__(self, info):
        self.basis, self.coeffs, self.centroid, _class_name, _, self.orientation, label = info
        self.class_name = _class_name[0]

    def __repr__(self):
        return 'BoundingBox3DV2 - class_name: {}, basis: {}, coeffs: {}, centroid: {}, orientation: {}'.format(
            self.class_name, self.basis, self.coeffs, self.centroid, self.orientation)

    def aabb(self, center_size: bool):
        centroid = self.centroid[0]  # (3, )
        basis = self.basis  # (3, 3)
        coeffs = self.coeffs[0]  # (3, )
        dx = basis[0] * coeffs[0]
        dy = basis[1] * coeffs[1]
        dz = basis[2] * coeffs[2]

        pl = [sx * dx + sy * dy + sz * dz for sx, sy, sz in CL]
        cx, cy, cz, sx, sy, sz = aabb_from_oriented_bbox(np.array(pl))

        if center_size:
            return cx + centroid[0], cy + centroid[1], cz + centroid[2], sx, sy, sz
        else:
            return [[cx + 0.5 * dx * sx + centroid[0],
                     cy + 0.5 * dy * sy + centroid[1],
                     cz + 0.5 * dz * sz + centroid[2]] for dx, dy, dz in CL]


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
        return 'seq_name: {}'.format(self.seq_root)

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

    def load_rgb(self):
        return np.asarray(cv2.cvtColor(cv2.imread(str(self.rgb_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
                          dtype=np.uint8)

    def load_depth(self):
        return np.asarray(cv2.imread(str(self.depth_path), cv2.IMREAD_UNCHANGED), dtype=np.uint16)


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


def compute_inverse_rt(extrinsics):
    """Computes the inverse 3x3 matrix from world coordinate to camera coordinate."""
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = extrinsics
    Tyz = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1]], dtype=np.float32)
    Rt = Tyz @ Rt
    iRt = np.linalg.inv(Rt)[:3, :3]
    return iRt


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

    @property
    def fx(self):
        return self.K[0, 0]

    @property
    def fy(self):
        return self.K[1, 1]

    @property
    def cx(self):
        return self.K[0, 2]

    @property
    def cy(self):
        return self.K[1, 2]

    def project_2d(self, p):
        return p[0] / p[2] * self.fx + self.cx, p[1] / p[2] * self.fy + self.cy

    def project_2d_bbox_from_3d_bbox(self, object_id: int, h: int = -1, w: int = -1):
        bbox_3d = self.gt_3d_bbox[object_id]
        inv_rt = compute_inverse_rt(self.extrinsics)

        cx_, cy_, cz_, sx_, sy_, sz_ = bbox_3d.aabb(center_size=True)
        y_ = cy_
        x1_, x2_ = cx_ - 0.5 * sx_, cx_ + 0.5 * sx_
        z1_, z2_ = cz_ - 0.5 * sz_, cz_ + 0.5 * sz_

        # Choose a plane with the mean y-value.
        # P1 +-----+ P2
        #    |     |
        # P4 +-----+ P3
        P1 = np.array((x1_, y_, z2_)) @ np.transpose(inv_rt)
        P2 = np.array((x2_, y_, z2_)) @ np.transpose(inv_rt)
        P3 = np.array((x2_, y_, z1_)) @ np.transpose(inv_rt)
        P4 = np.array((x1_, y_, z1_)) @ np.transpose(inv_rt)

        # Project the corner points into 2D plane.
        pp1 = self.project_2d(P1)
        pp2 = self.project_2d(P2)
        pp3 = self.project_2d(P3)
        pp4 = self.project_2d(P4)

        y1 = 0.5 * (pp1[1] + pp2[1])
        y2 = 0.5 * (pp3[1] + pp4[1])
        x1 = 0.5 * (pp1[0] + pp4[0])
        x2 = 0.5 * (pp2[0] + pp3[0])

        if w > 0 and h > 0:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
        # x, y, w, h
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


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
