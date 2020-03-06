import numpy as np
from PIL import Image
import random


# 利用pillow从路径中读取一张图片
def read_image(path, dtype=np.float32, color=True):
    f = Image.open(path)
    try:
        if color:   # 默认为三通道彩色图
            img = f.convert('RGB')
        else:   # 灰度矩阵
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:   # 若为灰度图，返回(1, H, W)矩阵
        return img[np.newaxis]
    else:   # (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


# 对图片按指定轴随机进行翻转
def random_flip(img, y_random=False, x_random=False,
                return_param=False, copy=False):
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}

    return img


# bbox坐标随图片尺寸变换
def resize_bbox(bbox, in_size, out_size):
    bbox = bbox.copy()
    y_scale = float(out_size[0] / in_size[0])
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]

    return bbox


# bbox随图片进行翻转
def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max

    return bbox
