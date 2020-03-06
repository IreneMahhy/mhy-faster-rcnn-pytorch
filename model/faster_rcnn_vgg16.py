from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN, RoIHead, normal_init
from utils.config import opt


def decom_vgg16():
    if opt.caffe_pretrain:  # 加载caffe pretrain参数
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.vgg_caffe_pretrain_path))
    else:   # 加载pytorch pretrain参数
        model = vgg16(not opt.load_path)

    # 去除最后的max pooling
    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]   # 删除最后一个fc层
    if not opt.use_drop:    # 删除dropout
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # 冻结前四个卷积层的参数 加快训练速度 不需要求梯度
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class FasterRCNNVGG16(FasterRCNN):
    feat_stride = 16

    def __init__(self, n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]):
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            fc7=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head
        )


class VGG16RoIHead(RoIHead):
    def __init__(self, n_class, roi_size, spatial_scale, fc7):
        # 用于分类和回归的fc层
        super(VGG16RoIHead, self).__init__(n_class, roi_size, spatial_scale, fc7)

        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

    def head_to_tail(self, pool):
        pool = pool.view(pool.size(0), -1)
        fc = self.fc7(pool)
        return fc
