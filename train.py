from __future__ import absolute_import
import os
import resource
# though cupy is not used but without this line, it raise errors...
import cupy as cp
from torch.utils import data as data_

import ipdb
from tqdm import tqdm
import matplotlib

from utils.config import opt
from utils import array_tool as at
from data.dataset import Dataset, TestDataset, inverse_normalize
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from model.faster_rcnn_res101 import FasterRCNNRes101
from trainer import FasterRCNNTrainer
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


# 从测试数据中选择10000个进行mAP评估
def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num:
            break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True
    )

    return result


def train(**kwargs):
    opt._parse(kwargs)  # 解析传入的参数
    dataset = Dataset(opt)
    print('load data... ...')
    # 分别载入训练集和测试集的数据
    # shuffle:每个epoch前是否对数据重新排序
    # num_workers:使用几个进程载入数据
    dataloader = data_.DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       shuffle=False,
                                       num_workers=opt.test_num_workers,
                                       pin_memory=True)

    if opt.pretrained_model == 'vgg16':
        faster_rcnn = FasterRCNNVGG16()
    elif opt.pretrained_model == 'res101':
        faster_rcnn = FasterRCNNRes101()
    print('Model construct completed!')

    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    # 数据集object类别列表
    trainer.vis.text(dataset.db.label_names, win='labels')

    # 开始训练
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

            # 预设每n次训练打印一次数据
            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                trainer.vis.plot_many(trainer.get_meter_data())

                # 将训练图片逆正则化后显示出来，并显示ground truth bbox
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_, at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # 对训练图片进行预测得到预测结果，并显示出来
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())

            # 测试集评估结果
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{}, loss:{}'.format(str(lr_),
                                                   str(eval_result['map']),
                                                   str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == opt.epoch:
            break


if __name__ == '__main__':
    import fire

    fire.Fire()

