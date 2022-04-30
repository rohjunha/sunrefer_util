import json
from typing import List

from utils.directory import fetch_intrinsic_from_tag_path, fetch_intrinsic_tag_by_image_id_path


class IntrinsicFetcher:
    def __init__(self):
        self.intrinsic_from_tag_path = fetch_intrinsic_from_tag_path()
        self.intrinsic_tag_by_image_id_path = fetch_intrinsic_tag_by_image_id_path()

        with open(str(self.intrinsic_from_tag_path), 'r') as file:
            self.intrinsic_from_tag = json.load(file)
        with open(str(self.intrinsic_tag_by_image_id_path), 'r') as file:
            self.intrinsic_tag_by_image_id = json.load(file)

    def __getitem__(self, image_id) -> List[float]:
        # fx, fy, x0, y0, h, w
        return self.intrinsic_from_tag[self.intrinsic_tag_by_image_id[image_id]]
