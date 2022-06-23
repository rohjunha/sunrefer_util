from pathlib import Path
from typing import Tuple

import cv2
import lmdb
import msgpack
import msgpack_numpy
import numpy as np
from tqdm import tqdm

from utils.directory import fetch_concise_storage_path, fetch_concise_rgb_path, fetch_concise_depth_path

msgpack_numpy.patch()


def encode_str(query: str) -> bytes:
    return query.encode()


def decode_str(byte_str: bytes) -> str:
    return byte_str.decode('utf-8')


class PointCloudStorage:
    def __init__(
            self,
            read_only: bool = True,
            db_path: Path = None):
        self.db_path = db_path
        self.num_dbs = 2
        self.env = lmdb.open(
            path=str(self.db_path),
            max_dbs=self.num_dbs,
            map_size=int(5e11),
            max_readers=1,
            readonly=read_only,
            lock=False,
            readahead=False,
            meminit=False
        )
        self.rgb = self.env.open_db(encode_str('rgb'))
        self.depth = self.env.open_db(encode_str('depth'))

    def _get_num_items(self, db) -> int:
        with self.env.begin(db=db, write=False) as txn:
            return txn.stat()['entries']

    def __len__(self):
        return self._get_num_items(self.rgb)

    def key(self, image_id: str):
        return encode_str(image_id)

    def _get_array(
            self,
            db,
            image_id: str) -> np.array:
        with self.env.begin(db=db, write=False) as txn:
            return msgpack.unpackb(txn.get(self.key(image_id)))

    def _put_array(
            self,
            db,
            image_id: str,
            item: np.array):
        with self.env.begin(db=db, write=True) as txn:
            txn.put(self.key(image_id), msgpack.packb(item))

    def get_rgb(self, image_id: str) -> np.array:
        return self._get_array(db=self.rgb, image_id=image_id)

    def get_depth(self, image_id: str) -> np.array:
        return self._get_array(db=self.depth, image_id=image_id)

    def get_rgb_and_depth(self, image_id: str) -> Tuple[np.array, np.array]:
        return self.get_rgb(image_id), self.get_depth(image_id)

    def put_rgb(self, image_id: str, rgb: np.array):
        self._put_array(db=self.rgb, image_id=image_id, item=rgb)

    def put_depth(self, image_id: str, depth: np.array):
        self._put_array(db=self.depth, image_id=image_id, item=depth)

    def __del__(self):
        self.env.close()


def fetch_point_cloud_storage() -> PointCloudStorage:
    storage_path = fetch_concise_storage_path()
    if not storage_path.exists():
        storage = PointCloudStorage(read_only=False, db_path=storage_path)
        for image_id in tqdm(['{:06d}'.format(i) for i in range(1, 10336)]):
            rgb = cv2.imread(str(fetch_concise_rgb_path(image_id)), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(fetch_concise_depth_path(image_id)), cv2.IMREAD_UNCHANGED)
            storage.put_rgb(image_id, rgb)
            storage.put_depth(image_id, depth)
    return PointCloudStorage(read_only=True, db_path=storage_path)


if __name__ == '__main__':
    storage = fetch_point_cloud_storage()
    rgb = storage.get_rgb('000001')
    depth = storage.get_depth('000001')
    print(rgb.shape, depth.dtype, np.max(depth))
