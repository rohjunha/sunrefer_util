import base64
import json
from pathlib import Path

import cv2
from tqdm import tqdm

from utils.directory import fetch_sunrefer_anno_path
from utils.intrinsic_fetcher import IntrinsicFetcher

intrinsic_fetcher = IntrinsicFetcher()


def create_tsv_from_storage_sunrefer(split: str):
    with open(str(fetch_sunrefer_anno_path()), 'r') as file:
        refer_by_bbox_id = json.load(file)

    dataset_dir = Path.home() / 'projects/ofa/dataset/sunrefer_data'
    idx_path = dataset_dir / '{}_data_idx.txt'.format(split)
    with open(str(idx_path), 'r') as file:
        image_id_set = set(map(lambda x: '{:06d}'.format(int(x)), file.read().splitlines()))

    image_dir = Path('/home/junha/data/sunrefer/xyzrgb_extrinsic')
    out_tsv = Path('/home/junha/data/sunrefer/sunrefer_{}.tsv'.format(split))

    with open(str(out_tsv), 'w') as tsv_file:
        for bbox_id, refer_item in list(refer_by_bbox_id.items()):
            image_id, object_id = bbox_id.split('_')
            # if image_id in {'005979', '005978'}:
            if image_id in image_id_set:
                fx, fy, cx, cy, h, w = intrinsic_fetcher[image_id]
                intrinsic_path = image_dir / '{}.json'.format(image_id)
                with open(str(intrinsic_path), 'r') as file:
                    intrinsic_dict = json.load(file)
                ncx, ncy = intrinsic_dict['cx'], intrinsic_dict['cy']
                offset_x = ncx - cx
                offset_y = ncy - cy

                # print(fx, fy, cx, cy, h, w)
                # print(intrinsic_dict)
                # print(offset_x, offset_y)
                bbox = refer_item['bbox2d']  # x1, y1, w, h
                bbox = [
                    int(offset_x + bbox[0]),
                    int(offset_y + bbox[1]),
                    int(offset_x + bbox[0] + bbox[2]),
                    int(offset_y + bbox[1] + bbox[3])]  # x1, y1, x2, y2

                # img = cv2.imread(str(image_dir / '{}.jpg'.format(image_id)))
                # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255))
                # cv2.imwrite('/home/junha/Downloads/{}.jpg'.format(image_id), img)
                # continue

                region_coords = ','.join([str(int(v)) for v in bbox])
                with open(str(image_dir / '{}.jpg'.format(image_id)), 'rb') as file:
                    image_str = base64.b64encode(file.read()).decode('utf-8')
                for i, sentence in enumerate(refer_item['sentences']):
                    anno_id = '{}_{}'.format(bbox_id, i)
                    out_str = '\t'.join([anno_id, sentence, region_coords, image_str]) + '\n'
                    tsv_file.write(out_str)


    # with open(str(out_tsv), 'w') as tsv_file:
    #     for bbox_id, refer_item in tqdm(refer_by_bbox_id.items()):
    #         image_id, object_id = bbox_id.split('_')
    #         if image_id in image_id_set:
    #             bbox = refer_item['bbox2d']  # x1, y1, w, h
    #             bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # x1, y1, x2, y2
    #             region_coords = ','.join([str(v) for v in bbox])
    #             with open(str(image_dir / '{}.jpg'.format(image_id)), 'rb') as file:
    #                 image_str = base64.b64encode(file.read()).decode('utf-8')
    #             # sentences = ';'.join(refer_item['sentences'])
    #             for i, sentence in enumerate(refer_item['sentences']):
    #                 anno_id = '{}_{}'.format(bbox_id, i)
    #                 out_str = '\t'.join([anno_id, sentence, region_coords, image_str]) + '\n'
    #                 tsv_file.write(out_str)


if __name__ == '__main__':
    create_tsv_from_storage_sunrefer('test')
