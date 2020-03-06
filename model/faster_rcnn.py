from __future__ import  absolute_import
from __future__ import division
import torch as t
import numpy as np
import cupy as cp
from torch import nn
from torch import functional as F

from utils import array_tool as at
from utils.config import opt
from data.dataset import preprocess
from model.utils.bbox_tools import loc2bbox
from model.utils.nms import non_maximum_suppression
from model.roi_module import RoIPooling2D


# 作为注释使函数忽略梯度
def nograd(f):
    def new_f(*args, **kwargs):
        with t.no_grad():
            return f(*args, **kwargs)
    return new_f

class FasterRCNN(nn.Module):
    """
    Args:
        extractor (nn.Module): A module that takes a BCHW image
            array and returns feature maps.
        rpn (nn.Module): A module that has the same interface as
            :class:`model.region_proposal_network.RegionProposalNetwork`.
            Please refer to the documentation found there.
        head (nn.Module): A module that takes
            a BCHW variable, RoIs and batch indices for RoIs. This returns class
            dependent localization paramters and class scores.
        loc_normalize_mean (tuple of four floats): Mean values of
            localization estimates.
        loc_normalize_std (tupler of four floats): Standard deviation
            of localization estimates.
    """
    def __init__(self, extractor, rpn, head,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('evaluate') # ???

    @property
    def n_class(self):
        # 包括背景在内的类别总数
        return self.head.n_class

    # 前向传播  x为4维的图片batch，scale即preprocess过程对原图的缩放
    def forward(self, x, scale=1.):
        img_size = x.shape[2:]

        fm = self.extractor(x)   # 得到feature map
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(fm, img_size, scale)
        roi_cls_locs, roi_scores = self.head(
            fm, rois, roi_indices)

        return roi_cls_locs, roi_scores, rois, roi_indices

    # 预测时指定preset，使用预设的阈值
    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')


    @nograd
    def predict(self, imgs, sizes=None, visualize=False):
        self.eval()    # 将training属性设为False，涉及BN和Dropout的更改
        if visualize:
            self.use_preset('visualize')
            prepared_imgs = list()
            sizes = list()
            # 对图片依次进行预处理
            for img in imgs:
                size = img.shape[1:]    # H W
                img = preprocess(at.tonumpy(img))
                prepared_imgs.append(img)
                sizes.append(size)
        else:
            prepared_imgs = imgs

        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = at.totensor(img[None]).float()    # 转换为tensor并增加一个维度
            scale = img.shape[3] / size[1]  # 预处理后W / 原W
            roi_cls_locs, roi_scores, rois, _ = self(img, scale)
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_locs.data
            roi = at.totensor(rois) / scale  # roi缩放回原图大小

            mean = t.Tensor(self.loc_normalize_mean).cuda(). \
                repeat(self.n_class)[None]      # (84,)
            std = t.Tensor(self.loc_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            # shape(N, 21, 4)
            cls_bbox = loc2bbox(at.tonumpy(roi).reshape((-1, 4)),
                                at.tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = at.totensor(cls_bbox).view(-1, self.n_class * 4)
            # 调整bbox后剪裁到图片范围内
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            # softmax计算各类别分数概率
            prob = at.tonumpy(t.softmax(at.totensor(roi_score), dim=1))

            cls_bbox = at.tonumpy(cls_bbox)
            prob = at.tonumpy(prob)

            # 对bbox再次进行阈值保留和nms
            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset('evaluate')
        self.train()    # 设为训练模式
        return bboxes, labels, scores

    def _suppress(self, cls_bbox, prob):
        bbox = list()
        label = list()
        score = list()

        # 每类分别根据分数阈值和nms过滤，保留符合条件的bbox和分数
        for i in range(1, self.n_class):
            cls_bbox_i = cls_bbox.reshape((-1, self.n_class, 4))[:, i, :]
            prob_i = prob[:, i]

            mask = prob_i > self.score_thresh
            cls_bbox_i = cls_bbox_i[mask]
            prob_i = prob_i[mask]

            keep = non_maximum_suppression(
                cp.array(cls_bbox_i), self.nms_thresh, prob_i)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_i[keep])
            label.append((i - 1) * np.ones((len(keep),)))
            score.append(prob_i[keep])

        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)

        return bbox, label, score

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer


class RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, fc7):
        super(RoIHead, self).__init__()

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)
        self.fc7 = fc7

    def forward(self, x, rois, roi_indices):
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # --> (ymin, ymax, xmin, xmax, indices)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        # vgg16分类器为线性层，res101模型layer4为Bottleneck
        fc = self.head_to_tail(pool)
        roi_cls_locs = self.cls_loc(fc)  # fc84
        roi_scores = self.score(fc)  # fc21

        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
