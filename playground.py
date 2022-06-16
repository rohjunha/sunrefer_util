import json
import math
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import List, Dict, Tuple, Any

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy.io
import torch
from PIL import Image
from tqdm import tqdm

from scripts.create_3d_bbox_dataset import verify_single_meta_file
from utils.directory import fetch_sunrefer_anno_path, fetch_xyzrgb_bbox_path, fetch_xyzrgb_pcd_path, \
    fetch_split_train_path, fetch_split_test_path, fetch_segformer_path, fetch_intrinsic_tag_by_image_id_path, \
    fetch_intrinsic_from_tag_path, fetch_official_sunrgbd_dir, fetch_xyzrgb_mask_path
from utils.intrinsic_fetcher import IntrinsicFetcher
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


def compute_volume(x1, x2, y1, y2, z1, z2):
    return torch.multiply(torch.multiply(x2 - x1, y2 - y1), z2 - z1)


def fetch_sides(centers, sizes):
    # centers, sizes: (3, ), (3, )
    x1 = centers[0] - 0.5 * sizes[0]
    x2 = centers[0] + 0.5 * sizes[0]
    y1 = centers[1] - 0.5 * sizes[1]
    y2 = centers[1] + 0.5 * sizes[1]
    z1 = centers[2] - 0.5 * sizes[2]
    z2 = centers[2] + 0.5 * sizes[2]
    return x1, x2, y1, y2, z1, z2


def compute_iou(ref_centers, ref_sizes, tar_centers, tar_sizes):
    ref_x1, ref_x2, ref_y1, ref_y2, ref_z1, ref_z2 = fetch_sides(ref_centers, ref_sizes)
    tar_x1, tar_x2, tar_y1, tar_y2, tar_z1, tar_z2 = fetch_sides(tar_centers, tar_sizes)

    if ref_x1 > ref_x2 or ref_y1 > ref_y2 or ref_z1 > ref_z2:
        print('invalid ref sides', ref_x1, ref_x2, ref_y1, ref_y2, ref_z1, ref_z2)
        return 0., 0.
    if tar_x1 > tar_x2 or tar_y1 > tar_y2 or tar_z1 > tar_z2:
        print('invalid tar sides', tar_x1, tar_x2, tar_y1, tar_y2, tar_z1, tar_z2)
        return 0., 0.

    v1 = compute_volume(ref_x1, ref_x2, ref_y1, ref_y2, ref_z1, ref_z2)
    v2 = compute_volume(tar_x1, tar_x2, tar_y1, tar_y2, tar_z1, tar_z2)

    int_x1 = torch.max(ref_x1, tar_x1)
    int_x2 = torch.min(ref_x2, tar_x2)
    int_y1 = torch.max(ref_y1, tar_y1)
    int_y2 = torch.min(ref_y2, tar_y2)
    int_z1 = torch.max(ref_z1, tar_z1)
    int_z2 = torch.min(ref_z2, tar_z2)
    if int_x1 > int_x2 or int_y1 > int_y2 or int_z1 > int_z2:
        vi = 0.
        # print('invalid intersection', int_x1, int_x2, int_y1, int_y2, int_z1, int_z2)
        # print('ref', ref_x1, ref_x2, ref_y1, ref_y2, ref_z1, ref_z2)
        # print('tar', tar_x1, tar_x2, tar_y1, tar_y2, tar_z1, tar_z2)
        # return 0., 0.
    else:
        vi = compute_volume(int_x1, int_x2, int_y1, int_y2, int_z1, int_z2)

    uni_x1 = torch.min(ref_x1, tar_x1)
    uni_x2 = torch.max(ref_x2, tar_x2)
    uni_y1 = torch.min(ref_y1, tar_y1)
    uni_y2 = torch.max(ref_y2, tar_y2)
    uni_z1 = torch.min(ref_z1, tar_z1)
    uni_z2 = torch.max(ref_z2, tar_z2)
    if uni_x1 > uni_x2 or uni_y1 > uni_y2 or uni_z1 > uni_z2:
        print('invalid union', uni_x1, uni_x2, uni_y1, uni_y2, uni_z1, uni_z2)
        return 0., 0.
    vu = compute_volume(uni_x1, uni_x2, uni_y1, uni_y2, uni_z1, uni_z2)

    v_union = v1 + v2 - vi
    if v_union < 0 or abs(v_union) < 1e-5:
        print('invalid v_union', v_union)

    iou = vi / v_union
    if math.isnan(iou):
        print('ref', ref_centers, ref_sizes)
        print('tar', tar_centers, tar_sizes)
        print('ref sides', ref_x1, ref_x2, ref_y1, ref_y2, ref_z1, ref_z2)
        print('tar sides', tar_x1, tar_x2, tar_y1, tar_y2, tar_z1, tar_z2)
        print('v1, v2, vi, v_union, vu', v1, v2, vi, v_union, vu)

    giou = iou - (vu - v_union) / (vu + 1e-7)
    return iou, giou


def compute_iou_from_xyz(x):
    xyz, tar_centers, tar_sizes = x
    bounds = torch.quantile(xyz, torch.tensor([0.2, 0.8]), dim=0)  # (2, 3)
    mask = (xyz > bounds[0, ...]) & (xyz < bounds[1, ...])  # (n, 3)
    mask = mask[..., 0] & mask[..., 1] & mask[..., 2]  # (n, )
    ref_centers = torch.divide(torch.sum(torch.multiply(xyz, mask.unsqueeze(-1)), dim=0),
                               torch.sum(mask, dim=0).view(-1, 1)).view(-1, )  # (b, 3)
    ref_sizes = (bounds[1, ...] - bounds[0, ...]) * 2  # (b, 3)
    iou, _ = compute_iou(ref_centers, ref_sizes, tar_centers, tar_sizes)
    return iou.item()


def prepare_eval_from_pcd_data(split: str, use_seg_out: bool, mask_ratio: float = 0.8):
    split_path = fetch_split_train_path() if split == 'train' else fetch_split_test_path()
    with open(str(split_path), 'r') as file:
        image_id_list = list(map(lambda x: '{:06d}'.format(int(x)), file.read().splitlines()))
    pred_bbox_dict: Dict[str, List[Tuple[str, List[float]]]] = fetch_predicted_bbox_by_image_id()
    with open(str(fetch_xyzrgb_bbox_path()), 'r') as file:
        aabb_dict: Dict[str, List[Tuple[str, List[float]]]] = json.load(file)

    item_list = []
    for image_id in tqdm(image_id_list):
        pcd = np.load(str(fetch_xyzrgb_pcd_path(image_id)))
        if use_seg_out:
            seg_out = np.load(str(fetch_segformer_path(image_id)))
            seg_out_mask = np.zeros(seg_out.shape, dtype=bool)
            for label in [0, 3, 5]:
                seg_out_mask[seg_out == label] = True
        for uniq_id, bbox_2d in pred_bbox_dict[image_id]:
            image_id, object_id, anno_id = uniq_id.split('_')
            object_id = int(object_id)
            x1, y1, w, h = list(map(int, bbox_2d))
            x2, y2 = x1 + w, y1 + h
            pcd_crop = pcd[y1:y2, x1:x2, :3]
            valid = (np.abs(pcd_crop[..., 0]) > 1e-5) | (np.abs(pcd_crop[..., 1]) > 1e-5) | \
                    (np.abs(pcd_crop[..., 2]) > 1e-5)  # (w, h)
            if use_seg_out:
                if np.sum(seg_out_mask[y1:y2, x1:x2].astype(np.int64)) < (y2 - y1) * (x2 - x1) * mask_ratio:
                    valid &= ~seg_out_mask[y1:y2, x1:x2]
            xyz = torch.from_numpy(pcd_crop[valid, :]).to(dtype=torch.float32)  # (n, 3)
            tar_bbox = torch.tensor(aabb_dict[image_id][object_id][1]).to(dtype=torch.float32)
            tar_centers, tar_sizes = tar_bbox[:3], tar_bbox[3:6]
            item_list.append((xyz, tar_centers, tar_sizes))
    return item_list


def eval_from_pcd_data(split: str, use_seg_out: bool):
    items = prepare_eval_from_pcd_data(split, use_seg_out)

    pool = Pool(12)
    iou = np.array(list(tqdm(pool.imap_unordered(compute_iou_from_xyz, items), total=len(items))))
    iou = iou[~np.isnan(iou)]
    print('mIoU     {:5.3f}'.format(np.mean(iou) * 100.))
    print('acc@0.25 {:5.3f}'.format(np.sum(iou > 0.25) / iou.shape[0] * 100))
    print('acc@0.5  {:5.3f}'.format(np.sum(iou > 0.5) / iou.shape[0] * 100))


class EvaluationFromPCD:
    def __init__(self):
        self.pred_bbox_dict: Dict[str, List[Tuple[str, List[float]]]] = fetch_predicted_bbox_by_image_id()
        with open(str(fetch_xyzrgb_bbox_path()), 'r') as file:
            self.aabb_dict: Dict[str, List[Tuple[str, List[float]]]] = json.load(file)

    def compute_iou_from_split(self, split: str):
        split_path = fetch_split_train_path() if split == 'train' else fetch_split_test_path()
        with open(str(split_path), 'r') as file:
            image_id_list = list(map(lambda x: '{:06d}'.format(int(x)), file.read().splitlines()))
        iou_list = []
        for image_id in tqdm(image_id_list):
            iou_list += self.compute_iou_from_image_id(image_id)
        iou = np.array(iou_list)

        print('mIoU     {:5.3f}'.format(np.mean(iou) * 100.))
        print('acc@0.25 {:5.3f}'.format(np.sum(iou > 0.25) / iou.shape[0] * 100))
        print('acc@0.5  {:5.3f}'.format(np.sum(iou > 0.5) / iou.shape[0] * 100))

    def compute_iou_from_image_id(self, image_id: str):
        iou_list = []
        pcd = np.load(str(fetch_xyzrgb_pcd_path(image_id)))
        for uniq_id, bbox_2d in self.pred_bbox_dict[image_id]:
            image_id, object_id, anno_id = uniq_id.split('_')
            object_id = int(object_id)
            x, y, w, h = list(map(int, bbox_2d))
            pcd_crop = pcd[y:y + h, x:x + w, :3]
            valid = (np.abs(pcd_crop[..., 0]) > 1e-5) | (np.abs(pcd_crop[..., 1]) > 1e-5) | (
                        np.abs(pcd_crop[..., 2]) > 1e-5)
            xyz = torch.from_numpy(pcd_crop[valid, :]).to(dtype=torch.float32)  # (n, 3)

            bounds = torch.quantile(xyz, torch.tensor([0.2, 0.8]), dim=0)  # (2, 3)
            mask = (xyz > bounds[0, ...]) & (xyz < bounds[1, ...])  # (n, 3)
            mask = mask[..., 0] & mask[..., 1] & mask[..., 2]  # (n, )
            ref_centers = torch.divide(torch.sum(torch.multiply(xyz, mask.unsqueeze(-1)), dim=0),
                                       torch.sum(mask, dim=0).view(-1, 1)).view(-1, )  # (b, 3)
            ref_sizes = (bounds[1, ...] - bounds[0, ...]) * 2  # (b, 3)
            tar_bbox = torch.tensor(self.aabb_dict[image_id][object_id][1]).to(dtype=torch.float32)
            tar_centers, tar_sizes = tar_bbox[:3], tar_bbox[3:6]
            iou, _ = compute_iou(ref_centers, ref_sizes, tar_centers, tar_sizes)
            iou_list.append(iou.item())
        return iou_list


def project_3d_to_2d_bbox():
    image_id = '000001'

    scene_by_image_id = fetch_scene_object_by_image_id('v2_3d')
    scene = scene_by_image_id[image_id]
    rgb = cv2.imread(str(scene.rgb_path), cv2.IMREAD_COLOR)
    w, h = rgb.shape[1], rgb.shape[0]
    pcd = np.load(str(fetch_xyzrgb_pcd_path(image_id)))[..., :5]  # x, y, z, rx, ry
    pcd[..., 3] *= w  # xx
    pcd[..., 4] *= h  # yy
    fx, fy = scene.K[0, 0], scene.K[1, 1]
    tx, ty = scene.K[0, 2], scene.K[1, 2]
    pcd = pcd[(np.abs(pcd[..., 0]) > 1e-5) | (np.abs(pcd[..., 1]) > 1e-5) | (np.abs(pcd[..., 2]) > 1e-5), :]
    pcd[:, 1] *= -1.
    pcd[:, 2] *= -1.
    pcd[:, :3] = pcd[:, :3] @ scene.extrinsics
    print(fx, fy, tx, ty, w, h, pcd.shape)

    for i in range(1000):
        X, Y, Z, xx_, yy_ = pcd[i, :]
        xx = X / Z * fx + tx
        yy = Y / Z * fy + ty
        print(i, xx_, xx, yy_, yy)

    # x = X / Z * fx + cx
    # y = Y / Z * fy + cy

    # with open(str(fetch_xyzrgb_bbox_path()), 'r') as file:
    #     aabb_dict: Dict[str, List[Tuple[str, List[float]]]] = json.load(file)
    # aabb_info = aabb_dict[image_id][4]
    # print(aabb_info)
    #
    # scene_by_image_id = fetch_scene_object_by_image_id('v2_3d')
    #
    # with open(str(fetch_sunrefer_anno_path()), 'r') as file:
    #     anno_dict = json.load(file)
    # for anno_key, anno_item in list(anno_dict.items())[:1]:
    #     image_id, object_id = anno_key.split('_')
    #     bbox_2d = anno_item['bbox2d']
    #     object_name = anno_item['object_name']
    #     sentences = anno_item['sentences']
    #     object_id = int(object_id)
    #
    #     scene = scene_by_image_id[image_id]
    #     fx = scene.K[0, 0]
    #     fy = scene.K[1, 1]
    #     tx = scene.K[0, 2]
    #     ty = scene.K[1, 2]
    #
    #     cx, cy, cz, sx, sy, sz = aabb_info[image_id][object_id][1]
    #     print(image_id, object_id, object_name, bbox_2d)
    #
    #
    #
    #
    #
    #
    #
    # scene = scene_by_image_id[image_id]
    # image = cv2.cvtColor(cv2.imread(str(scene.rgb_path)), cv2.COLOR_BGR2RGB)
    # bbox = scene.gt_2d_bbox[int(object_id)]
    # x1, y1, w, h = bbox.gt_bbox_2d
    # cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
    # plt.imshow(image)
    # plt.show()

    # print(scene)
    # print(scene.gt_2d_bbox[4])

    # X = (x - cx) * Z / fx
    # Y = (y - cy) * Z / fy
    # x = X / Z * fx + cx
    # y = Y / Z * fy + cy


INTRINSIC_BY_TAG = {
    "k0": [518.857901, 519.469611, 284.582449, 208.736166, 427, 561],
    "k1": [520.532, 520.7444, 277.9258, 215.115, 427, 561],
    "k2": [529.5, 529.5, 365.0, 265.0, 530, 730],
    "x0": [570.342224, 570.342224, 310.0, 225.0, 441, 591],
    "x1": [533.069214, 533.069214, 310.0, 225.0, 441, 591],
    "x2": [570.342224, 570.342224, 291.0, 231.0, 441, 591],
    "r0": [691.584229, 691.584229, 362.777557, 264.75, 531, 681],
    "r1": [693.74469, 693.74469, 360.431915, 264.75, 531, 681]}


class SceneInformation:
    def __init__(self, item_dict: Dict[str, Any]):
        self.fx, self.fy, self.tx, self.ty, self.height, self.width = INTRINSIC_BY_TAG[item_dict['intrinsic']]
        self._item_dict = item_dict
        self.seq_name = item_dict['seq_name']
        self.rgb_name = item_dict['rgb_name']
        self.depth_name = item_dict['depth_name']
        self.E = item_dict['E']
        self.obj_3d_list = item_dict['obj_3d']
        self.obj_2d_list = item_dict['obj_2d']

    @property
    def seq_dir(self):
        return fetch_official_sunrgbd_dir() / self.seq_name

    @property
    def depth_path(self) -> Path:
        if self.seq_dir is not None and self.depth_name is not None:
            return self.seq_dir / 'depth' / self.depth_name
        else:
            return Path()

    @property
    def rgb_path(self) -> Path:
        if self.seq_dir is not None and self.rgb_name is not None:
            return self.seq_dir / 'image' / self.rgb_name
        else:
            return Path()

    @property
    def seg_path(self) -> Path:
        if self.seq_dir is not None:
            return self.seq_dir / 'seg.mat'
        else:
            return Path()


def fetch_scene_object_by_image_id_revised() -> Dict[str, SceneInformation]:
    converted_scene_info_path = Path('/home/junha/data/sunrefer/meta.pt')
    if converted_scene_info_path.exists():
        converted_scene_info_by_image_id = torch.load(str(converted_scene_info_path))
    else:
        scene_3d_by_image_id = fetch_scene_object_by_image_id('v2_3d')
        scene_2d_by_image_id = fetch_scene_object_by_image_id('v2_2d')

        with open(str(fetch_intrinsic_tag_by_image_id_path()), 'r') as file:
            intrinsic_tag_by_image_id = json.load(file)

        converted_scene_info_by_image_id = dict()
        for key in tqdm(scene_3d_by_image_id.keys()):
            scene_3d = scene_3d_by_image_id[key]
            scene_2d = scene_2d_by_image_id[key]
            intrinsic_tag = intrinsic_tag_by_image_id[key]

            obj_3d_list = []
            obj_2d_list = []

            for object_index, bbox_3d in enumerate(scene_3d.gt_3d_bbox):
                obj_3d_list.append({
                    'class_name': bbox_3d.class_name,
                    'basis': bbox_3d.basis,
                    'coeffs': bbox_3d.coeffs,
                    'centroid': bbox_3d.centroid,
                    'orientation': bbox_3d.orientation})

            for object_index, bbox_2d in enumerate(scene_2d.gt_2d_bbox):
                obj_2d_list.append({
                    'class_name': bbox_2d.class_name,
                    'has_3d_bbox': bbox_2d.has_3d_bbox,
                    'bbox_2d': bbox_2d.gt_bbox_2d})

            converted_scene_info_by_image_id[key] = {
                'seq_name': scene_3d.seq_root.stem,
                'rgb_name': scene_3d.rgb_name,
                'depth_name': scene_3d.depth_name,
                'E': scene_3d.extrinsics,
                'intrinsic': intrinsic_tag,
                'obj_3d': obj_3d_list,
                'obj_2d': obj_2d_list,
            }

        torch.save(converted_scene_info_by_image_id, converted_scene_info_path)
    return {k: SceneInformation(v) for k, v in converted_scene_info_by_image_id.items()}


def inlier_ratio_from_pcd_and_aabb(pcd, aabb):
    X, Y, Z, W, H, D = aabb
    X1, X2 = X - 0.5 * W, X + 0.5 * W
    Y1, Y2 = Y - 0.5 * H, Y + 0.5 * H
    Z1, Z2 = Z - 0.5 * D, Z + 0.5 * D
    inliers = (X1 <= pcd[:, 0]) & (pcd[:, 0] <= X2) & \
              (Y1 <= pcd[:, 1]) & (pcd[:, 1] <= Y2) & \
              (Z1 <= pcd[:, 2]) & (pcd[:, 2] <= Z2)
    return np.sum(inliers) / pcd.shape[0]


def compute_2d_3d_bbox_correspondences():
    MIN_NUM_POINTS = 20
    MIN_INLIER_RATIO = 0.2

    scene_object_by_image_id = fetch_scene_object_by_image_id_revised()
    bbox3d_by_image_id: Dict[str, List[Tuple[str, List[float]]]] = json.load(open(str(fetch_xyzrgb_bbox_path()), 'r'))
    obj_pair_list = []
    for image_id, aabb_list in tqdm(list(bbox3d_by_image_id.items())):
        pcd = np.load(str(fetch_xyzrgb_pcd_path(image_id)))  # H, W, 8
        mask = (pcd[:, :, 0] < 1e-5) & (pcd[:, :, 1] < 1e-5) & (pcd[:, :, 2] < 1e-5)
        scene = scene_object_by_image_id[image_id]

        aabb_dict = defaultdict(list)
        for class_name_3d, aabb in aabb_list:
            aabb_dict[class_name_3d].append(aabb)

        for obj_2d in scene.obj_2d_list:
            class_name_2d = obj_2d['class_name']
            has_bbox_3d = obj_2d['has_3d_bbox']
            x, y, w, h = obj_2d['bbox_2d']

            if not has_bbox_3d:
                continue

            if len(aabb_dict[class_name_2d]) > 0:
                mask_crop = mask[y:y + h, x:x + w]
                pcd_crop = pcd[y:y + h, x:x + w, :3]
                pcd_crop = pcd_crop[~mask_crop, :]  # N, 3
                num_points = pcd_crop.shape[0]
                if num_points < MIN_NUM_POINTS:
                    continue

                sampled_xyz = pcd_crop
                centroid = np.mean(sampled_xyz, axis=0)
                dist_list = [np.linalg.norm(centroid - aabb[:3]) for class_name, aabb in aabb_list]
                irl = [(inlier_ratio_from_pcd_and_aabb(sampled_xyz, aabb), aabb)
                       for aabb in aabb_dict[class_name_2d]]
                irl = sorted(irl, key=lambda x: x[0], reverse=True)
                best_inlier_ratio, best_aabb = irl[0]
                if best_inlier_ratio > MIN_INLIER_RATIO:
                    obj_pair_list.append((image_id, class_name_2d, obj_2d['bbox_2d'], best_aabb))
                # else:
                #     ious, _ = zip(*irl)
                #     print('{}, {}: could not find the aabb larger than iou of 0.2: {}'.format(
                #         image_id, class_name_2d, list(ious)))
            # else:
            #     irl = [(class_name_3d, inlier_ratio_from_pcd_and_aabb(sampled_xyz, aabb))
            #            for class_name_3d, aabb in aabb_list]
            #     print(image_id, class_name_2d, dist_list)
            #     print(image_id, class_name_2d, irl)
            #     print('{} was not found from 3d'.format(class_name_2d))


        # for class_name, aabb in aabb_list:
        #     print(image_id, class_name, aabb)

    with open('/home/junha/data/sunrefer/obj_pair_info.json', 'w') as file:
        json.dump(obj_pair_list, file, indent=4)


def test_2d_3d_bbox_correspondences():
    with open('/home/junha/data/sunrefer/obj_pair_info.json', 'r') as file:
        obj_pair_list = json.load(file)
    print(len(obj_pair_list))

    with open('/home/junha/data/sunrefer/object_2d_3d.json', 'r') as file:
        obj_2d_3d = json.load(file)
    print(sum(len(items) for image_id, items in obj_2d_3d.items()))


class BoundingBoxCorrespondenceDataset:
    def __init__(self, num_points: int):
        with open('/home/junha/data/sunrefer/obj_pair_info.json', 'r') as file:
            self.obj_pair_list = json.load(file)
        self.num_points = num_points

    def __len__(self):
        return len(self.obj_pair_list)

    def __getitem__(self, item):
        image_id, class_name, (x, y, w, h), bbox_3d = self.obj_pair_list[item]
        pcd = np.load(str(fetch_xyzrgb_pcd_path(image_id)))  # H, W, 8
        mask = np.load(str(fetch_xyzrgb_mask_path(image_id)))  # H, W
        mask_crop = mask[y:y + h, x:x + w]
        pcd_crop = pcd[y:y + h, x:x + w, :]
        pcd_crop = pcd_crop[~mask_crop, :]  # N, 8
        pcd_sample = pcd_crop[np.random.choice(pcd_crop.shape[0], self.num_points), :]
        return pcd_sample, bbox_3d


def create_masks():
    for i in tqdm(range(1, 10336)):
        image_id = '{:06d}'.format(i)
        pcd = np.load(str(fetch_xyzrgb_pcd_path(image_id)))  # H, W, 8
        mask = (pcd[:, :, 0] < 1e-5) & (pcd[:, :, 1] < 1e-5) & (pcd[:, :, 2] < 1e-5)
        np.save(str('/home/junha/data/sunrefer/xyzrgb/mask{}.npy'.format(image_id)), mask)


def compute_iou_from_gt_3d_bbox():
    pred_bbox_by_image_id = fetch_predicted_bbox_by_image_id()
    scene_by_image_id = fetch_scene_object_by_image_id_revised()
    aabb_list_by_image_id = json.load(open(str(fetch_xyzrgb_bbox_path()), 'r'))
    anno_dict = json.load(open(str(fetch_sunrefer_anno_path()), 'r'))
    iou_list = []
    for image_id, pred_bbox_list in tqdm(pred_bbox_by_image_id.items()):
        scene = scene_by_image_id[image_id]
        aabb_list = aabb_list_by_image_id[image_id]
        pcd = np.load(str(fetch_xyzrgb_pcd_path(image_id)))  # H, W, 8
        mask = np.load(str(fetch_xyzrgb_mask_path(image_id)))  # H, W
        for uniq_id, bbox_2d in pred_bbox_list:
            _, object_id, anno_id = uniq_id.split('_')
            image_object_id = '{}_{}'.format(image_id, object_id)
            gt_class_name = anno_dict[image_object_id]['object_name']
            object_id = int(object_id)
            x, y, w, h = list(map(int, bbox_2d))
            x = max(0, min(scene.width - 1, x))
            y = max(0, min(scene.height - 1, y))
            w = max(0, min(scene.width, w))
            h = max(0, min(scene.height, h))
            mask_crop = mask[y:y + h, x:x + w]
            pcd_crop = pcd[y:y + h, x:x + w, :]
            pcd_crop = pcd_crop[~mask_crop, :]  # N, 8

            irl = [(inlier_ratio_from_pcd_and_aabb(pcd_crop, aabb), class_name, aabb) for class_name, aabb in aabb_list]
            irl = sorted(irl, key=lambda x: x[0], reverse=True)
            _, prd_class_name, prd_bbox_3d = irl[0]
            _, tgt_bbox_3d = aabb_list[object_id]
            prd_bbox_3d = torch.tensor(prd_bbox_3d)
            tgt_bbox_3d = torch.tensor(tgt_bbox_3d)
            iou, _ = compute_iou(tgt_bbox_3d[:3], tgt_bbox_3d[3:], prd_bbox_3d[:3], prd_bbox_3d[3:])
            iou_list.append(iou.item())
            # print('tgt_class: {}'.format(gt_class_name))
            # print('prd_class: {}'.format(pred_class_name))
            # print('tgt_bbox: {}'.format(aabb_list[object_id]))
            # print('prd_bbox: {}'.format(pred_bbox_3d))
        # break

    iou = np.array(iou_list)
    print('number of predictions: {}'.format(iou.shape[0]))
    print('mean iou: {:5.3f}'.format(np.mean(iou) * 100.))
    print('acc@0.25: {:5.3f}'.format(np.sum(iou > 0.25) / iou.shape[0] * 100.))
    print('acc@0.5 : {:5.3f}'.format(np.sum(iou > 0.5) / iou.shape[0] * 100.))


if __name__ == '__main__':
    # visualize_2d_bbox('000951')
    # detect_wall_and_floor('002920')
    # test_sampled_dataset()
    # create_sampled_points('train', 1000)
    # create_sampled_points('test', 1000)
    # evaluate_with_xyzrgb()
    # eval_from_pcd_data(split='val', use_seg_out=True)
    # project_3d_to_2d_bbox()
    # scene_by_image_id = test_fetch_scene_object_by_image_id()
    # scene = scene_by_image_id['000001']
    # print(scene.tx)
    # verify_single_meta_file()
    # compute_2d_3d_bbox_correspondences()
    # test_2d_3d_bbox_correspondences()
    # compute_2d_3d_bbox_correspondences()
    # dataset = BoundingBoxCorrespondenceDataset(num_points=1000)
    # for pcd, bbox in tqdm(dataset):
    #     pass
    # create_masks()
    compute_iou_from_gt_3d_bbox()
