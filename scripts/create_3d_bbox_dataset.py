import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.directory import fetch_split_train_path, fetch_split_test_path, fetch_object_2d_3d_path, \
    fetch_xyzrgb_pcd_path, fetch_xyzrgb_bbox_path, fetch_object_pair_path, fetch_xyzrgb_mask_path, fetch_segformer_path
from utils.intrinsic_fetcher import IntrinsicFetcher
from utils.meta_io import MetaObject2DV2


class BoundingBoxEstimationDataset(Dataset):
    def __init__(self, split: str, num_samples: int):
        Dataset.__init__(self)
        self.split = split
        self.num_samples = num_samples
        self.idx_path = fetch_split_train_path() if split == 'train' else fetch_split_test_path()
        with open(str(self.idx_path), 'r') as file:
            self.image_id_list = list(map(lambda x: '{:06d}'.format(int(x)), file.read().splitlines()))
        with open(str(fetch_object_2d_3d_path()), 'r') as file:
            object_2d_3d_dict = json.load(file)

        self.items = []
        for image_id in self.image_id_list:
            items = object_2d_3d_dict[image_id]
            for class_name, idx_2d, bbox_2d, idx_3d, bbox_3d in items:
                self.items.append((image_id, bbox_2d, bbox_3d))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_id, bbox_2d, bbox_3d = self.items[index]
        x, y, w, h = bbox_2d
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        xyz = np.load(str(fetch_xyzrgb_pcd_path(image_id)))
        mask = np.logical_or(np.logical_or(np.abs(xyz[:, :, 0]) > 1e-5, np.abs(xyz[:, :, 1]) > 1e-5),
                             np.abs(xyz[:, :, 2]) > 1e-5)
        mask_ = mask[y:y + h, x:x + w]
        xyz_ = xyz[y:y + h, x:x + w, :3][mask_, :]
        if xyz_.shape[0] == 0:
            print(image_id, bbox_2d)
            return None, None
        sampled_xyz = xyz_[np.random.choice(xyz_.shape[0], self.num_samples), :]
        return sampled_xyz, bbox_3d


class SampledBoundingBoxEstimationDataset(Dataset):
    def __init__(self, split: str):
        Dataset.__init__(self)
        self.split = split
        self.items = torch.load('/home/junha/data/sunrefer/pcd_{}.pt'.format(split))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # class_name, sampled_xyz, bbox_3d = self.items[index]
        return self.items[index]


def create_sampled_points(split: str, num_samples: int):
    idx_path = fetch_split_train_path() if split == 'train' else fetch_split_test_path()
    with open(str(idx_path), 'r') as file:
        image_id_list = list(map(lambda x: '{:06d}'.format(int(x)), file.read().splitlines()))
    # with open(str(fetch_object_2d_3d_path()), 'r') as file:
    #     object_2d_3d_dict = json.load(file)
    with open(str(fetch_object_pair_path()), 'r') as file:
        object_pair_list = json.load(file)

    object_pairs_by_image_id = defaultdict(list)
    for image_id, cls_name, bbox_2d, bbox_3d in object_pair_list:
        object_pairs_by_image_id[image_id].append((cls_name, bbox_2d, bbox_3d))

    output = []
    for image_id in tqdm(image_id_list):
        pcd = np.load(str(fetch_xyzrgb_pcd_path(image_id)))
        mask = np.load(str(fetch_xyzrgb_mask_path(image_id)))
        items = object_pairs_by_image_id[image_id]
        height, width = mask.shape[0], mask.shape[1]
        # items = object_2d_3d_dict[image_id]
        # mask = np.logical_or(np.logical_or(np.abs(xyz[:, :, 0]) > 1e-5, np.abs(xyz[:, :, 1]) > 1e-5),
        #                      np.abs(xyz[:, :, 2]) > 1e-5)

        for cls_name, bbox_2d, bbox_3d in items:
            x, y, w, h = bbox_2d
            x1 = max(0, min(width - 1, x))
            y1 = max(0, min(height - 1, y))
            w = max(0, min(width, w))
            h = max(0, min(height, h))
            x2 = max(0, min(width, x1 + w))
            y2 = max(0, min(height, x2 + h))
            mask_ = mask[y1:y2, x1:x2]  # (H, W)
            pcd_ = pcd[y1:y2, x1:x2, :][mask_, :]  # (N, 8)
            if pcd_.shape[0] < 20:
                continue
            sampled_pcd = pcd_[np.random.choice(pcd_.shape[0], num_samples), :]
            output.append((cls_name, sampled_pcd, bbox_3d))

        # for class_name, idx_2d, bbox_2d, idx_3d, bbox_3d in items:
        #     x, y, w, h = bbox_2d
        #     if x < 0:
        #         x = 0
        #     if y < 0:
        #         y = 0
        #     mask_ = mask[y:y + h, x:x + w]
        #     xyz_ = xyz[y:y + h, x:x + w, :][mask_, :]
        #     if xyz_.shape[0] == 0:
        #         continue
        #     sampled_xyz = xyz_[np.random.choice(xyz_.shape[0], num_samples), :]
        #     output.append((class_name, sampled_xyz, bbox_3d))
    torch.save(output, '/home/junha/data/sunrefer/pcd_{}.pt'.format(split))


def create_sampled_points_revised(
        split: str,
        num_samples: int):
    idx_path = fetch_split_train_path() if split == 'train' else fetch_split_test_path()
    with open(str(idx_path), 'r') as file:
        image_id_list = list(map(lambda x: '{:06d}'.format(int(x)), file.read().splitlines()))
    with open(str(fetch_object_pair_path()), 'r') as file:
        object_pair_list = json.load(file)
    intrinsic_fetcher = IntrinsicFetcher()

    object_list_by_image_id = defaultdict(list)
    for image_id, cls_name, bbox_2d, bbox_3d in object_pair_list:
        object_list_by_image_id[image_id].append((cls_name, bbox_2d, bbox_3d))

    output = []
    for image_id in tqdm(image_id_list):
        _, _, _, _, height, width = intrinsic_fetcher[image_id]
        pcd = np.load(str(fetch_xyzrgb_pcd_path(image_id)))
        mask = np.load(str(fetch_xyzrgb_mask_path(image_id)))
        assert height == mask.shape[0]
        assert width == mask.shape[1]
        items = object_list_by_image_id[image_id]
        for cls_name, bbox_2d, bbox_3d in items:
            x, y, w, h = bbox_2d
            x1 = max(0, min(width - 1, x))
            y1 = max(0, min(height - 1, y))
            w = max(0, min(width, w))
            h = max(0, min(height, h))
            x2 = max(0, min(width, x1 + w))
            y2 = max(0, min(height, y1 + h))

            mask_crop = mask[y1:y2, x1:x2]
            pcd_crop = pcd[y1:y2, x1:x2, :][mask_crop, :]
            if pcd_crop.shape[0] < 20:
                continue
            pcd_sampled = pcd_crop[np.random.choice(pcd_crop.shape[0], num_samples), :]
            output.append((cls_name, pcd_sampled, bbox_3d))

    print('saving {} items'.format(len(output)))
    torch.save(output, '/home/junha/data/sunrefer/pcd_{}.pt'.format(split))


def create_object_pair_with_segformer_mask():
    obj_by_image_id = torch.load('/home/junha/data/sunrefer/meta.pt')

    def center_size_from_bbox_dict(bbox_dict):
        CL = [
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1]]

        """Applies swapping (z, -y) to the final bbox to fit with xyz data."""
        centroid = np.array(bbox_dict['centroid'])[0]  # (3, )
        basis = np.array(bbox_dict['basis'])  # (3, 3)
        coeffs = np.array(bbox_dict['coeffs'])[0]  # (3, )
        ux = basis[0] * coeffs[0]
        uy = basis[1] * coeffs[1]
        uz = basis[2] * coeffs[2]
        verts = np.array([sx * ux + sy * uy + sz * uz for sx, sy, sz in CL])
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
        cs = np.array([cx, cy, cz, sx, sy, sz])
        cs[:3] += centroid
        # return cs
        new_cs = np.array([cs[0], cs[2], -cs[1], cs[3], cs[5], cs[4]])
        return new_cs

    res = dict()
    for image_id, scene in tqdm(obj_by_image_id.items()):
        matched_bbox_list = []
        candidate_bbox_3d_list = []

        obj_3d_list = scene['obj_3d']
        class_dict = defaultdict(list)
        for bbox_2d in scene['obj_2d']:
            class_dict[bbox_2d['class_name']].append(bbox_2d['bbox_2d'])  # (x, y, w, h)

        for i, obj_3d in enumerate(obj_3d_list):
            class_name = obj_3d['class_name']
            if class_name not in class_dict:
                continue

            bbox_3d = center_size_from_bbox_dict(obj_3d)
            if len(class_dict[class_name]) > 1:
                candidate_bbox_3d_list.append((class_name, bbox_3d))
            else:
                bbox_2d = class_dict[class_name][0]
                matched_bbox_list.append((class_name, bbox_2d, bbox_3d.tolist()))

        if candidate_bbox_3d_list:
            pcd = np.load(str(fetch_xyzrgb_pcd_path(image_id)))
            mask = np.load(str(fetch_xyzrgb_mask_path(image_id)))
            seg_out = np.load(str(fetch_segformer_path(image_id)))
            seg_out_mask = np.zeros(seg_out.shape, dtype=bool)
            for label in [0, 3, 5]:
                seg_out_mask[seg_out == label] = True

            for class_name, bbox_3d in candidate_bbox_3d_list:
                bbox_2d_list = class_dict[class_name]
                dist_bbox_list = []
                for x1, y1, w, h in bbox_2d_list:
                    x2, y2 = x1 + w, y1 + h
                    pcd_crop = pcd[y1:y2, x1:x2, :3]  # (h, w, 3)
                    mask_crop = mask[y1:y2, x1:x2]  # (h, w)
                    seg_out_crop = seg_out_mask[y1:y2, x1:x2]  # (h, w)
                    eff_mask_crop = mask_crop & ~seg_out_crop
                    pcd_eff = pcd_crop[eff_mask_crop, :]
                    num_points = pcd_eff.shape[0]
                    if num_points < 10:
                        continue
                    sampled_xyz = pcd_eff[np.random.choice(num_points, 1000), :]
                    centroid = np.mean(sampled_xyz, axis=0)
                    dist = np.linalg.norm(centroid - bbox_3d[:3])
                    dist_bbox_list.append((dist, (x1, y1, w, h)))
                dist_bbox_list = sorted(dist_bbox_list, key=lambda x: x[0])
                if not dist_bbox_list:
                    continue
                dist, bbox_2d = dist_bbox_list[0]
                if dist > 1.:
                    print('{}::too far from the gt center: {}'.format(image_id, dist))
                else:
                    matched_bbox_list.append((class_name, bbox_2d, bbox_3d.tolist()))

        res[image_id] = matched_bbox_list

    with open('/home/junha/data/sunrefer/object_2d_3d_segformer.json', 'w') as file:
        json.dump(res, file, indent=4)


def create_sampled_points_segformer(
        split: str,
        num_samples: int):
    idx_path = fetch_split_train_path() if split == 'train' else fetch_split_test_path()
    with open(str(idx_path), 'r') as file:
        image_id_list = list(map(lambda x: '{:06d}'.format(int(x)), file.read().splitlines()))
    with open('/home/junha/data/sunrefer/object_2d_3d_segformer.json', 'r') as file:
        object_pair_list_by_image_id = json.load(file)
    intrinsic_fetcher = IntrinsicFetcher()

    output = []
    for image_id in tqdm(image_id_list):
        _, _, _, _, height, width = intrinsic_fetcher[image_id]
        pcd = np.load(str(fetch_xyzrgb_pcd_path(image_id)))
        mask = np.load(str(fetch_xyzrgb_mask_path(image_id)))
        seg_out = np.load(str(fetch_segformer_path(image_id)))
        seg_out_mask = np.zeros(seg_out.shape, dtype=bool)
        for label in [0, 3, 5]:
            seg_out_mask[seg_out == label] = True

        assert height == mask.shape[0]
        assert width == mask.shape[1]
        items = object_pair_list_by_image_id[image_id]
        for cls_name, bbox_2d, bbox_3d in items:
            x, y, w, h = bbox_2d
            x1 = max(0, min(width - 1, x))
            y1 = max(0, min(height - 1, y))
            w = max(0, min(width, w))
            h = max(0, min(height, h))
            x2 = max(0, min(width, x1 + w))
            y2 = max(0, min(height, y1 + h))

            mask_crop = mask[y1:y2, x1:x2]
            seg_out_crop = seg_out_mask[y1:y2, x1:x2]
            eff_mask_crop = mask_crop & ~seg_out_crop
            pcd_crop = pcd[y1:y2, x1:x2, :][eff_mask_crop, :]
            if pcd_crop.shape[0] < 20:
                continue
            pcd_sampled = pcd_crop[np.random.choice(pcd_crop.shape[0], num_samples), :]
            output.append((cls_name, pcd_sampled, bbox_3d))

    print('saving {} items'.format(len(output)))
    torch.save(output, '/home/junha/data/sunrefer/pcd/pcd_1000_seg/pcd_{}.pt'.format(split))


def test_sampled_dataset():
    import time
    dataset = SampledBoundingBoxEstimationDataset(split='train')
    count = 0
    t1 = time.time()
    for _, xyz, bbox in tqdm(dataset):
        count += 1
    t2 = time.time()
    print(count, t2 - t1)


def create_verified_obj_2d_3d_correspondences(split: str):
    idx_path = fetch_split_train_path() if split == 'train' else fetch_split_test_path()
    with open(str(idx_path), 'r') as file:
        image_id_list = list(map(lambda x: '{:06d}'.format(int(x)), file.read().splitlines()))
    with open(str(fetch_object_2d_3d_path()), 'r') as file:
        object_list_by_image_id = json.load(file)

    object_pairs_by_image_id = defaultdict(list)
    for image_id, obj_info_list in object_list_by_image_id.items():
        for cls_name, i1, bbox_2d, i2, bbox_3d in obj_info_list:
            object_pairs_by_image_id[image_id].append((cls_name, bbox_2d, bbox_3d))

    output = []
    for image_id in tqdm(image_id_list):
        pcd = np.load(str(fetch_xyzrgb_pcd_path(image_id)))
        mask = np.load(str(fetch_xyzrgb_mask_path(image_id)))
        items = object_pairs_by_image_id[image_id]
        height, width = mask.shape[0], mask.shape[1]

        for cls_name, bbox_2d, bbox_3d in items:
            x, y, w, h = bbox_2d
            x1 = max(0, min(width - 1, x))
            y1 = max(0, min(height - 1, y))
            w = max(0, min(width, w))
            h = max(0, min(height, h))
            x2 = max(0, min(width, x1 + w))
            y2 = max(0, min(height, x2 + h))
            mask_crop = mask[y1:y2, x1:x2]  # (H, W)
            pcd_crop = pcd[y1:y2, x1:x2, :][mask_crop, :]  # (N, 8)
            if pcd_crop.shape[0] < 30:
                continue
            output.append((image_id, cls_name, bbox_2d, bbox_3d))
    torch.save(output, '/home/junha/data/sunrefer/verified_obj_2d_3d_{}.pt'.format(split))


if __name__ == '__main__':
    # create_object_pair_with_segformer_mask()
    create_sampled_points_segformer('train', 1000)
    # create_sampled_points_segformer('test', 1000)
    create_verified_obj_2d_3d_correspondences('train')
    create_verified_obj_2d_3d_correspondences('test')
