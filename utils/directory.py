from pathlib import Path


def _mkdir(p: Path) -> Path:
    if not p.exists():
        p.mkdir(parents=True)
    return p


def _ensure_exists(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError('could not find the file: {}'.format(p))
    return p


def fetch_data_root_dir() -> Path:
    return _ensure_exists(Path.home() / 'data/sunrefer')


def fetch_official_sunrgbd_dir() -> Path:
    return _ensure_file_exists('official')


def _ensure_file_exists(filename: str) -> Path:
    return _ensure_exists(fetch_data_root_dir() / filename)


def fetch_xyzrgb_dir() -> Path:
    """
    Returns a directory contains an extrinsic transformed pointcloud (x, y, z, rx, ry, r, g, b)
    in a shape of the image (W, H, 8).
    :return: a Path object to the xyzrgb directory.
    """
    return _ensure_file_exists('xyzrgb')


def fetch_xyzrgb_pcd_path(image_id: str) -> Path:
    return _ensure_exists(fetch_xyzrgb_dir() / '{}.npy'.format(image_id))


def fetch_xyzrgb_mask_path(image_id: str) -> Path:
    return _ensure_exists(fetch_xyzrgb_dir() / '{}_mask.npy'.format(image_id))


def fetch_xyzrgb_bbox_path() -> Path:
    return _ensure_exists(fetch_xyzrgb_dir() / 'aabb.json')


def fetch_rgb_dir() -> Path:
    return _ensure_file_exists('rgb')


def fetch_depth_data_dir() -> Path:
    return _ensure_file_exists('depth_data')


def fetch_seg_dir() -> Path:
    return _ensure_file_exists('seg')


def fetch_segformer_dir() -> Path:
    """
    Returns a directory contains segformer segmentation masks.
    :return: a Path object to the segformer directory.
    """
    return _ensure_file_exists('segformer')


def fetch_segformer_path(image_id: str) -> Path:
    return _ensure_exists(fetch_segformer_dir() / '{}.npy'.format(image_id))


def fetch_object_2d_3d_path() -> Path:
    return _ensure_file_exists('object_2d_3d.json')


def fetch_object_pair_path() -> Path:
    return _ensure_file_exists('obj_pair_info.json')


def fetch_sunrefer_anno_path() -> Path:
    return _ensure_file_exists('SUNREFER_v2_revised.json')


def fetch_intrinsic_from_tag_path() -> Path:
    return _ensure_file_exists('SUNREFER_intrinsic_from_tag.json')


def fetch_intrinsic_tag_by_image_id_path() -> Path:
    return _ensure_file_exists('SUNREFER_intrinsic_tag_by_image_id.json')


def fetch_train_tsv_path() -> Path:
    return _ensure_file_exists('sunrefer_train.tsv')


def fetch_test_tsv_path() -> Path:
    return _ensure_file_exists('sunrefer_test.tsv')


def fetch_val_tsv_path() -> Path:
    return _ensure_file_exists('sunrefer_val.tsv')


def fetch_ofa_predict_path() -> Path:
    return _ensure_file_exists('test_predict.json')


def fetch_split_train_path() -> Path:
    return _ensure_file_exists('train_data_idx.txt')


def fetch_split_test_path() -> Path:
    return _ensure_file_exists('test_data_idx.txt')


def fetch_split_val_path() -> Path:
    return _ensure_file_exists('val_data_idx.txt')


def fetch_meta_v1_path() -> Path:
    return _ensure_file_exists('SUNRGBDMeta.mat')


def fetch_meta_2d_path() -> Path:
    return _ensure_file_exists('SUNRGBDMeta2DBB_v2.mat')


def fetch_meta_3d_path() -> Path:
    return _ensure_file_exists('SUNRGBDMeta3DBB_v2.mat')


def fetch_rgb_path(image_id: str) -> Path:
    return _ensure_exists(fetch_rgb_dir() / '{}.jpg'.format(image_id))


def fetch_depth_data_path(image_id: str) -> Path:
    return _ensure_exists(fetch_depth_data_dir() / '{}.mat'.format(image_id))


def fetch_seg_path(image_id: str) -> Path:
    return _ensure_exists(fetch_seg_dir() / '{}.png'.format(image_id))


def fetch_concise_data_dir() -> Path:
    return _ensure_file_exists('xyzrgb_concise')


def fetch_concise_storage_path() -> Path:
    return _ensure_exists(fetch_concise_data_dir() / 'pcd')


def fetch_concise_meta_path() -> Path:
    return _ensure_exists(fetch_concise_data_dir() / 'meta.pt')


def fetch_concise_rgb_path(image_id: str) -> Path:
    return _ensure_exists(fetch_concise_data_dir() / '{}.jpg'.format(image_id))


def fetch_concise_depth_path(image_id: str) -> Path:
    return _ensure_exists(fetch_concise_data_dir() / '{}.png'.format(image_id))


def fetch_concise_refer_path() -> Path:
    return _ensure_exists(fetch_concise_data_dir() / 'sunrefer.json')


def fetch_concise_correspondence_path() -> Path:
    return _ensure_exists(fetch_concise_data_dir() / 'correspondence.json')


def fetch_concise_item_list_path() -> Path:
    return _ensure_exists(fetch_concise_data_dir() / 'item_list.txt')


def test_fetch_functions():
    fetch_data_root_dir()
    fetch_rgb_dir()
    fetch_depth_data_dir()
    fetch_seg_dir()
    fetch_sunrefer_anno_path()
    fetch_intrinsic_from_tag_path()
    fetch_intrinsic_tag_by_image_id_path()
    fetch_train_tsv_path()
    fetch_test_tsv_path()
    fetch_val_tsv_path()
    fetch_ofa_predict_path()
    fetch_split_train_path()
    fetch_split_test_path()
    fetch_split_val_path()
    fetch_meta_2d_path()
    fetch_meta_3d_path()
    for i in range(1, 6):
        image_id = '{:06d}'.format(i)
        fetch_rgb_path(image_id)
        fetch_depth_data_path(image_id)
        fetch_seg_path(image_id)


if __name__ == '__main__':
    test_fetch_functions()
