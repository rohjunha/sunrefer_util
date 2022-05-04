import json
from collections import defaultdict
from functools import partial
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.directory import fetch_sunrefer_anno_path, fetch_xyzrgb_bbox_path, fetch_xyzrgb_pcd_path, \
    fetch_split_train_path, fetch_split_test_path, fetch_object_2d_3d_path
from utils.intrinsic_fetcher import IntrinsicFetcher
from utils.line_mesh import LineMesh
from utils.meta_io import fetch_scene_object_by_image_id, MetaObject2DV2
from utils.pred_and_anno import fetch_predicted_bbox_by_image_id


def detect_wall_and_floor(image_id: str):
    with open('/home/junha/Downloads/objectinfo150.txt', 'r') as file:
        lines = file.read().splitlines()
    class_name_dict = dict()
    for l in lines[1:]:
        idx, ratio, train, val, fullname = l.split('\t')
        name = fullname.split(',')[0].strip()
        class_name_dict[int(idx) - 1] = name, fullname
    colors = scipy.io.loadmat('/home/junha/Downloads/color150.mat')['colors']

    test_dir = Path('/home/junha/projects/Refer-it-in-RGBD/sunrgbd/sunrgbd_trainval/image')
    out_dir = Path('/home/junha/Downloads/segformer')
    in_image_path = test_dir / '{}.jpg'.format(image_id)
    out_index_path = out_dir / '{}.npy'.format(image_id)
    assert in_image_path.exists()
    assert out_index_path.exists()

    image = Image.open(str(in_image_path))
    indices = np.load(str(out_index_path))

    color_image = np.zeros((indices.shape[0], indices.shape[1], 3), dtype=np.uint8)
    existing_label_indices = [0, 3, 5]
    for index in existing_label_indices:
        print('working on {}'.format(index))
        mask = indices == index
        if np.count_nonzero(mask) > 0:
            color_image += mask[:, :, np.newaxis] * np.tile(colors[index], (indices.shape[0], indices.shape[1], 1))

    fig = plt.figure(figsize=(30, 20))
    patch_list = []
    for color_index in existing_label_indices:
        patch_list.append(mpatches.Patch(color=colors[color_index].astype(np.float32) / 255.,
                                         label=class_name_dict[color_index][0]))

    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(image)
    ax.set_title('Input image')
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(color_image)
    ax.set_title('Segmentation')
    ax.legend(handles=patch_list)
    plt.show()


# def visualize_2d_bbox(image_id: str):
    scene_by_image_id = fetch_scene_object_by_image_id('v2_3d')
    pred_bbox_by_image_id = fetch_predicted_bbox_by_image_id()
    with open(str(fetch_sunrefer_anno_path()), 'r') as file:
        anno_data = json.load(file)
    anno_by_uniq_id = dict()
    for image_object_id, item in anno_data.items():
        sentences = item['sentences']
        bbox_gt = item['bbox2d']
        for anno_id, sentence in enumerate(sentences):
            uniq_id = '{}_{}'.format(image_object_id, anno_id)
            anno_by_uniq_id[uniq_id] = sentence, bbox_gt

    scene = scene_by_image_id[image_id]
    id_bbox_list = pred_bbox_by_image_id[image_id]
    rgb = cv2.imread(str(scene.rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(scene.depth_path), cv2.IMREAD_UNCHANGED)
    rgbd = o3d.geometry.RGBDImage.create_from_sun_format(
        o3d.geometry.Image(rgb), o3d.geometry.Image(depth), convert_rgb_to_intensity=False)

    # plt.subplot(1, 2, 1)
    # plt.title('SUN grayscale image')
    # plt.imshow(rgbd.color)
    # plt.subplot(1, 2, 2)
    # plt.title('SUN depth image')
    # plt.imshow(rgbd.depth)
    # plt.show()
    # print(np.min(rgbd.depth), np.max(rgbd.depth))
    # return

    for uniq_id, bbox in id_bbox_list:
        image_id, object_id, anno_id = uniq_id.split('_')


        color = np.asarray(rgbd.color)
        depth = np.asarray(rgbd.depth)

        x, y, w, h = bbox
        cv2.rectangle(color,
                      (int(x), int(y)), (int(x + w), int(y + h)),
                      color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        sentence, bbox_gt = anno_by_uniq_id[uniq_id]
        x, y, w, h = bbox_gt
        cv2.rectangle(color,
                      (int(x), int(y)), (int(x + w), int(y + h)),
                      color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        # cv2.rectangle(depth,
        #               (int(x), int(y)), (int(x + w), int(y + h)),
        #               color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        fig = plt.figure(figsize=(30, 20))
        fig.suptitle('Predicted bounding box from "{}"'.format(sentence))
        ax = fig.add_subplot(1, 2, 1)
        ax.set_title('SUN color image')
        plt.imshow(color)
        ax = fig.add_subplot(1, 2, 2)
        ax.set_title('SUN depth image')
        plt.imshow(depth)
        plt.show()


def build_single_meta_file():
    # scene2d_by_image_id = fetch_scene_object_by_image_id('v2_2d')
    # torch.save(scene2d_by_image_id, '/home/junha/Downloads/scene2d.pt')
    # scene3d_by_image_id = fetch_scene_object_by_image_id('v2_3d')
    # torch.save(scene3d_by_image_id, '/home/junha/Downloads/scene3d.pt')
    # return

    scene2d_by_image_id: Dict[str, MetaObject2DV2] = torch.load('/home/junha/Downloads/scene2d.pt')
    bbox3d_by_image_id: Dict[str, List[Tuple[str, List[float]]]] = json.load(open(str(fetch_xyzrgb_bbox_path()), 'r'))
    res = dict()
    for image_id, scene2d in list(scene2d_by_image_id.items()):
        bbox3d_list = bbox3d_by_image_id[image_id]
        class_dict = scene2d.build_class_dict()

        object_2d_and_3d_list = []
        examine_3d_list = []
        for i, bbox3d in enumerate(bbox3d_list):
            class_name, coordinates_3d = bbox3d
            if class_name not in class_dict:
                print('{}::invalid 3d bbox: {}, {}'.format(image_id, i, class_name))
                # print(list(map(lambda x: x.class_name, scene2d.gt_2d_bbox)))
                continue
            if len(class_dict[class_name]) > 1:
                examine_3d_list.append((i, bbox3d))
            else:
                bbox2d = class_dict[class_name][0]
                object_2d_and_3d_list.append((class_name, bbox2d.object_id, bbox2d.gt_bbox_2d, i, coordinates_3d))

        if examine_3d_list:
            xyzrgb_pcd = np.load(str(fetch_xyzrgb_pcd_path(image_id)))
            for i, bbox3d in examine_3d_list:
                class_name, coordinates_3d = bbox3d
                bbox2d_list = class_dict[class_name]
                dist_bbox_list = []
                for bbox2d in bbox2d_list:
                    x, y, w, h = bbox2d.gt_bbox_2d
                    xyz = xyzrgb_pcd[y:y + h, x:x + w, :3]
                    mask = np.logical_and(
                        np.logical_and(np.abs(xyz[:, :, 0]) < 1e-5,
                                       np.abs(xyz[:, :, 1]) < 1e-5),
                        np.abs(xyz[:, :, 2]) < 1e-5)
                    effective_xyz = xyz[~mask, :]
                    num_points = effective_xyz.shape[0]
                    if num_points < 10:
                        continue
                    sampled_xyz = effective_xyz[np.random.choice(num_points, 1000), :]
                    centroid = np.mean(sampled_xyz, axis=0)
                    dist = np.linalg.norm(centroid - coordinates_3d[:3])
                    dist_bbox_list.append((dist, bbox2d))
                dist_bbox_list = sorted(dist_bbox_list, key=lambda x: x[0])
                if not dist_bbox_list:
                    continue
                dist, bbox2d = dist_bbox_list[0]
                if dist > 1.:
                    print('{}::too far from the gt center: {}'.format(image_id, dist))
                else:
                    object_2d_and_3d_list.append((class_name, bbox2d.object_id, bbox2d.gt_bbox_2d, i, coordinates_3d))

        object_2d_and_3d_list = sorted(object_2d_and_3d_list, key=lambda x: x[1])
        res[image_id] = object_2d_and_3d_list

    with open('/home/junha/Downloads/object_2d_3d.json', 'w') as file:
        json.dump(res, file, indent=4)


def verify_single_meta_file():
    with open('/home/junha/Downloads/object_2d_3d.json', 'r') as file:
        object_2d_3d = json.load(file)
    for image_id, items in list(object_2d_3d.items())[:10]:
        print(image_id)
        xyz = np.load(str(fetch_xyzrgb_pcd_path(image_id)))
        print(items)
        for class_name, obj_id_2d, bbox_2d, obj_id_3d, bbox_3d in items:
            x, y, w, h = bbox_2d
            mask = np.logical_or(np.logical_or(np.abs(xyz[:, :, 0]) > 1e-5, np.abs(xyz[:, :, 1]) > 1e-5),
                                 np.abs(xyz[:, :, 2]) > 1e-5)
            mask_crop = mask[y:y+h, x:x+w]
            dx = xyz[y:y+h, x:x+w, 3][mask_crop] * xyz.shape[1]
            dy = xyz[y:y+h, x:x+w, 4][mask_crop] * xyz.shape[0]
            in_dx = np.logical_and(dx >= x, dx <= x + w)
            in_dy = np.logical_and(dy >= y, dy <= y + h)
            in_dxdy = np.logical_and(in_dx, in_dy)
            print(class_name, 'inlier rate: {:5.3f}'.format(np.sum(in_dxdy) / in_dx.shape[0] * 100))


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
        mask_ = mask[y:y+h, x:x+w]
        xyz_ = xyz[y:y+h, x:x+w, :3][mask_, :]
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
            xyz_ = xyz[y:y + h, x:x + w, :3][mask_, :]
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


if __name__ == '__main__':
    # visualize_2d_bbox('000951')
    # detect_wall_and_floor('002920')
    test_sampled_dataset()
