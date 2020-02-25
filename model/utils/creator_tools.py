import numpy as np
import cupy as cp

from .bbox_tools import loc2bbox, bbox_iou, bbox2loc
from .nms.non_maximum_suppression import non_maximum_suppression


class ProposalCreator:
    # 生成RoI
    # 参数：nms阈值，训练和测试时NMS前后保留的高分bbox数目，保留bbox的最小尺寸
    def __init__(self, parent_model, nms_thresh=0.7,
                 n_train_pre_nms=12000, n_train_post_nms=2000,
                 n_test_pre_nms=6000, n_test_post_nms=300,
                 min_size=16):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor,
                 img_size, scale=1.):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        roi = loc2bbox(anchor, loc)
        # 剪裁bbox使其不超过图片范围
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1])

        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]  # 求所有bbox的高和宽
        ws = roi[:, 3] - roi[:, 1]
        # np.where中条件给几维向量就输出几个array，每个array指定一个维度的坐标
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # 返回从大到小的分数索引，NMS之前首先选取适当数量
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]

        # 经过NMS处理返回合格的序号，再从中选取规定数量
        keep = non_maximum_suppression(
            cp.ascontiguousarray(cp.asarray(roi)),
            thresh=self.nms_thresh)
        if n_post_nms:
            keep = keep[:n_post_nms]
        roi = roi[keep]

        return roi


# 训练roi分类，选择128个正负RoI样本进行训练
class ProposalTargetCreator(object):
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        """"
        Args:
            roi (array): 待取样的RoI array (R, 4)
            bbox (array): gt bbox (R, 4)
            label (array): gt label (R,)
            loc_normalize_mean (tuple of four floats): bbox坐标的正则化mean值
            loc_normalize_std (tupler of four floats): bbox坐标的正则化标准差

        Returns:(array, array, array)
            * **sample_roi**: 取样的RoI (S, 4)
            * **gt_roi_loc**: 样本与gt间的坐标偏差 (S, 4)
            * **gt_roi_label**: RoI样本被指定的label（包括背景） (S,)   
        """
        n_bbox, _ = bbox.shape
#        roi = np.concatenate((roi, bbox), axis=0)
        pos_roi_per_img = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)  # 表示与roi交集最大的gt的索引
        max_iou = iou.max(axis=1)   # 表示每个roi与gt的iou最大值
        gt_roi_label = label[gt_assignment] + 1  # 表示与roi交集最大的gt的label

        # 求出所有归为正样本的roi序号，总数大于指定数目按指定选取，反之按总数选取
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image =  int(min(pos_roi_per_img, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index,
                                         pos_roi_per_this_image,
                                         replace=False)

        neg_index = np.where(max_iou < self.neg_iou_thresh_hi) & \
                            (max_iou > self.neg_iou_thresh_lo)
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index,
                                         neg_roi_per_this_image,
                                         replace=False)
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]     # 样本的训练目标label
        gt_roi_label[pos_roi_per_this_image:] = 0   # 负样本label为0即background
        sample_roi = roi[keep_index]    # 保留的训练样本roi

        # 求出roi与gt bbox间的偏差用于回归
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label


# 训练rpn，从所有Anchor中选择正负样本各128个进行二分类和回归
# IOU最高及>0.7的为正样本，IOU<0.3为负样本 只计算前景损失
class AnchorTargetCreator(object):
    def __init__(self, n_sample=256, pos_ratio=0.5,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh

    def __call__(self, bbox, anchor, img_size):
        img_H, img_W = img_size
        n_anchor = len(anchor)
        # 过滤超出图片范围的anchor
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        # 根据gt bbox给anchor赋label
        argmax_ious, label = self._create_label(anchor, bbox)

        # 每个anchor与对应有最大IoU的gt回归
        loc = bbox2loc(anchor, bbox[argmax_ious])

        # 将label和loc映射到过滤前的原始anchor集上
        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label


    def _create_label(self, anchor, bbox):
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        ious = bbox_iou(anchor, bbox)
        # 每个anchor对应IoU最大的gt序号
        argmax_ious = ious.argmax(ious, axis=1)
        max_ious = ious[np.arange(ious.shape[0]), argmax_ious]
        # 每个gt对应IoU最大的anchor序号
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        # anchor中max_iou小于阈值的为背景，max_iou大于阈值以及gt对应的最大anchor为前景
        label[max_ious < self.neg_iou_thresh] = 0
        label[gt_argmax_ious] = 1
        label[max_ious >= self.pos_iou_thresh] = 1

        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        # 选出一定数目的正样本和负样本，其他设为无关样本，label为-1
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos),
                replace=False)
            label[disable_index] = -1

        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label


def _get_inside_index(anchor, H, W):
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


#
def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret

