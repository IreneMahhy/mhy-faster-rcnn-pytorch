import numpy as np
import six


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    # 求中心点
    py = base_size / 2
    px = base_size / 2
    anchor_base = np.zeros((len(ratios)*len(anchor_scales), 4),
                           dtype=np.float32)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            # 由中心点求各个box的ymin, xmin, ymax, xmax坐标
            idx = i * len(anchor_scales) + j
            anchor_base[idx, 0] = py - h / 2
            anchor_base[idx, 1] = px - w / 2
            anchor_base[idx, 2] = py + h / 2
            anchor_base[idx, 3] = px + w / 2

    return anchor_base


def loc2bbox(src_bbox, loc):
    # 安全性检查
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)
    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    # 将源box的坐标转换为yxhw形式
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    ctr_y = loc[:, 0::4] * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = loc[:, 1::4] * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    height = np.exp(loc[:, 2::4]) * src_height[:, np.newaxis]
    width = np.exp(loc[:, 3::4]) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    # 切片时保持维度
    dst_bbox[:, 0::4] = ctr_y - 0.5 * height
    dst_bbox[:, 1::4] = ctr_x - 0.5 * width
    dst_bbox[:, 2::4] = ctr_y + 0.5 * height
    dst_bbox[:, 3::4] = ctr_x + 0.5 * width

    return dst_bbox


# 给定两个bbox计算其间的偏差
def bbox2loc(src_bbox, dst_bbox):
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


# 求两组bbox两两间的IoU，输入两组bbox分别为N维和K维，返回(N,K)矩阵
def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        return IndexError

    # 计算两个bbox相交部分的左上和右下点的坐标，利用广播增加一个维度
    # tl, br的维度为(N, K, 2)
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    # 若不相交，tl坐标<br坐标，则相交面积取0
    area_inter = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)

    return area_inter / (area_a[:, None] + area_b - area_inter)