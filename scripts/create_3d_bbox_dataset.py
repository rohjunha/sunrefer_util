import json

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.directory import fetch_split_train_path, fetch_split_test_path, fetch_object_2d_3d_path, \
    fetch_xyzrgb_pcd_path


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
    with open(str(fetch_object_2d_3d_path()), 'r') as file:
        object_2d_3d_dict = json.load(file)

    output = []
    for image_id in tqdm(image_id_list):
        xyz = np.load(str(fetch_xyzrgb_pcd_path(image_id)))
        items = object_2d_3d_dict[image_id]
        mask = np.logical_or(np.logical_or(np.abs(xyz[:, :, 0]) > 1e-5, np.abs(xyz[:, :, 1]) > 1e-5),
                             np.abs(xyz[:, :, 2]) > 1e-5)
        for class_name, idx_2d, bbox_2d, idx_3d, bbox_3d in items:
            x, y, w, h = bbox_2d
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            mask_ = mask[y:y + h, x:x + w]
            xyz_ = xyz[y:y + h, x:x + w, :][mask_, :]
            if xyz_.shape[0] == 0:
                continue
            sampled_xyz = xyz_[np.random.choice(xyz_.shape[0], num_samples), :]
            output.append((class_name, sampled_xyz, bbox_3d))
    torch.save(output, '/home/junha/data/sunrefer/pcd_{}.pt'.format(split))


def test_sampled_dataset():
    import time
    dataset = SampledBoundingBoxEstimationDataset(split='train')
    count = 0
    t1 = time.time()
    for _, xyz, bbox in tqdm(dataset):
        count += 1
    t2 = time.time()
    print(count, t2 - t1)
