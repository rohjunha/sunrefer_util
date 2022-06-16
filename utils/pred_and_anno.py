import json
from collections import defaultdict

from utils.directory import fetch_ofa_predict_path, fetch_sunrefer_anno_path


def fetch_predicted_bbox_by_uniq_id():
    with open(str(fetch_ofa_predict_path()), 'r') as file:
        pred_data = json.load(file)
    pred_bbox_by_uniq_id = dict()
    for pred_item in pred_data:
        bbox = pred_item['box']
        pred_bbox_by_uniq_id[pred_item['uniq_id']] = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
    return pred_bbox_by_uniq_id


def fetch_predicted_bbox_by_image_id():
    with open(str(fetch_ofa_predict_path()), 'r') as file:
        pred_data = sorted(json.load(file), key=lambda x:x['uniq_id'])
    pred_bbox_by_image_id = defaultdict(list)
    for pred_item in pred_data:
        uniq_id = pred_item['uniq_id']
        image_id, object_id, anno_id = uniq_id.split('_')
        bbox = pred_item['box']
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        pred_bbox_by_image_id[image_id].append((uniq_id, bbox))
    return pred_bbox_by_image_id


def fetch_sunrefer_anno_class_dict():
    with open(str(fetch_sunrefer_anno_path()), 'r') as file:
        sunrefer_anno_data = json.load(file)

    sunrefer_anno_class_dict = dict()
    for image_object_id, anno_dict in sunrefer_anno_data.items():
        for i in range(5):
            uniq_id = '{}_{}'.format(image_object_id, i)
            sunrefer_anno_class_dict[uniq_id] = anno_dict['object_name']
    return sunrefer_anno_class_dict
