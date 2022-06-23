import numpy as np


def convert_depth_float_from_uint8(raw_depth: np.ndarray):
    assert raw_depth.dtype == np.uint16
    depth = np.bitwise_or(np.right_shift(raw_depth, 3), np.left_shift(raw_depth, 13)).astype(np.float32) / 1000.
    depth[depth > 7.0] = 0.
    return depth


def unproject(float_depth, fx, fy, cx, cy):
    mask = float_depth > 0.
    mx = np.linspace(0, float_depth.shape[1] - 1, float_depth.shape[1], dtype=np.float32)
    my = np.linspace(0, float_depth.shape[0] - 1, float_depth.shape[0], dtype=np.float32)
    rx = np.linspace(0., 1., float_depth.shape[1], dtype=np.float32)
    ry = np.linspace(0., 1., float_depth.shape[0], dtype=np.float32)
    rx, ry = np.meshgrid(rx, ry)
    xx, yy = np.meshgrid(mx, my)
    xx = (xx - cx) * float_depth / fx
    yy = (yy - cy) * float_depth / fy
    pcd = np.stack((xx, yy, float_depth), axis=-1)
    return pcd, mask, rx, ry


def transform(pcd, E):
    return pcd @ np.transpose(E)
