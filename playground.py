import json
from collections import defaultdict
from functools import partial
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import List

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy.io
from PIL import Image
from tqdm import tqdm

from utils.directory import fetch_sunrefer_anno_path
from utils.intrinsic_fetcher import IntrinsicFetcher
from utils.line_mesh import LineMesh
from utils.meta_io import fetch_scene_object_by_image_id
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


if __name__ == '__main__':
    # visualize_2d_bbox('000951')
    detect_wall_and_floor('002920')
