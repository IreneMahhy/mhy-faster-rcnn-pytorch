import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F

from model.utils.bbox_tools import generate_anchor_base
from model.utils.creator_tools import ProposalCreator
from utils.config import opt


class RegionProposalNetwork(nn.Module):
    # ratios：anchor的宽高比；anchor_scales：anchor的面积
    # feat_stride：提取特征后的总stride；intialW：初始权重
    # proposal_creator_params：ProposalCreator的关键参数
    def __init__(self, in_channels=512, mid_channels=512,
                 ratios=[0.5, 1, 2], anchor_scales=[8,16,32],
                 feat_stride=16, proposal_creator_params=dict()
    ):
        super(RegionProposalNetwork, self).__init__()
        # 首先生成anchor_base
        self.anchor_base = generate_anchor_base(
            ratios=ratios, anchor_scales=anchor_scales)
        self.feat_stride = feat_stride
        # **kwargs即将字典作为关键字参数传递进函数
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]

        # vgg与resnet101的extractor输出维度不同
        if opt.pretrained_model == 'res101':
            in_channels = 1024

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)  # 3*3卷积
        # 分类任务卷积通道数为anchor*2，回归任务为anchor*4
        self.score = nn.Conv2d(mid_channels, n_anchor*2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor*4, 1, 1, 0)

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, hh, ww = x.shape  # n = batch_size
        # 在9个base_anchor基础上生成hh*ww*9个anchor，对应到原图坐标
        anchor = _enumerate_shifted_anchor(
            np.array(self.anchor_base), self.feat_stride, hh, ww)
        n_anchor = anchor.shape[0] // (hh * ww)

        h = F.relu(self.conv1(x))
        # 对分数和loc的预测仅通过卷积，没有激活层
        rpn_locs = self.loc(h)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()

        rpn_softmax_score = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_score[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)   # 每个样本所有anchor前景的概率
        rpn_scores = rpn_scores.view(n, -1, 2)

        # 生成RoI proposal
        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size, scale=scale)
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index) # batch中每个样本roi的batch序号
        # list concat为ndarray
        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor



def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    shift_y = np.arange(0, height*feat_stride, feat_stride)
    shift_x = np.arange(0, width*feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # shift(K,4) 四列为一个点的坐标*2，方便处理为box的两点坐标
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)
    A = anchor_base.shape[0]    # anchor_base数量一般为9
    K = shift.shape[0]  # 锚点的数量
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K*A, 4)).astype(np.float32)

    return anchor


# 正态初始化参数
def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        # 产生截断正态分布随机数，取值范围为 [ mean - 2 * stddev, mean + 2 * stddev ]
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)    # 并非精确估计 fmod为除法取余
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()