from __future__ import  absolute_import
from __future__ import  division
import torch as t
import numpy as np
from data import util
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from utils.config import opt


def preprocess(img, min_size=600, max_size=1000):
    '''短边缩放至self.min_size或长边取self.max_size，
    比较两个scale 看哪个才是主要影响缩放因子。如果是
    scale1更小，缩放完毕后那么将图片缩放到600*Z时Z
    不会超过1000，如果scale2更小，那么缩放到 Z*1000时，
    Z不会超过600，也即长边不能超过1000 短边不能超过600，
    且至少有一边是600或1000。最后图片规范化，整体减去自身中值self.mean。'''

    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    # 将矩阵值规范到0-1
    img = img / 255.
    # 使用skimage.transform中的resize方法缩放图片
    img = sktsf.resize(img, (C, H*scale, W*scale), mode='reflect', anti_aliasing=False)

    # 调用pytorch_normalze或者caffe_normalze对图像进行正则化
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalize

    return normalize(img)


# 标准化图片，即使其均值为0，方差为1
def pytorch_normalize(img):
    normalize = tvtsf.Normalize(mean=[], std=[])    # 调用torchvision.transforms
    img = normalize(t.from_numpy(img))

    return img.numpy()

# 将图片变为caffe标准，BGR，0-255，无标准差
def caffe_normalize(img):
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)

    return img


# 逆标准化图片，以便显示
def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[[2, 1, 0], :, :]
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))  # 变换ori_img

        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):  # 返回引用数据集的长度
        return len(self.db)


# 读取测试数据  split读取的是test.txt 启用use_difficult 不对图片 box等进行resize等处理
class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)

        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):    # 接收传入的元组
        img, bbox, label = in_data
        _, H, W = img.shape
        # 按大小范围（默认600~1000）等比例缩放图像，计算scale并调整bbox
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # 随机对图像进行水平翻转，并随之变化bbox
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale