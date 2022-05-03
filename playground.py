import json
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


def compute_iou_from_pool_func(pool, func, items):
    result_list_tqdm = []
    for result in tqdm(pool.imap_unordered(func, items), total=len(items)):
        result_list_tqdm.append(result)
    iou_by_uniq_id = {k: v for k, v in result_list_tqdm}

    with open('/home/junha/data/sunrefer/iou_by_uniq_id.json', 'w') as file:
        json.dump(iou_by_uniq_id, file, indent=4)

    iou_25 = 0
    iou_50 = 0
    for iou in iou_by_uniq_id.values():
        if iou > 0.25:
            iou_25 += 1
        if iou > 0.5:
            iou_50 += 1
    iou_25 = iou_25 / len(iou_by_uniq_id) * 100
    iou_50 = iou_50 / len(iou_by_uniq_id) * 100
    print('acc@0.25: {:5.3f}'.format(iou_25))
    print('acc@0.50: {:5.3f}'.format(iou_50))


if __name__ == '__main__':
    detect_wall_and_floor('002920')
