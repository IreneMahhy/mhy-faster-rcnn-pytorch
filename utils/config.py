from pprint import pprint


# 用于训练的参数设置
class Config:
    # data
    voc_data_dir = '/data/mahaoyun/voc/VOCdevkit/VOC2007/'
    min_size = 600
    max_size = 1000
    num_workers = 8
    test_num_workers = 8

    # sigma for L1_smooth_loss
    rpn_sigma = 3
    roi_sigma = 1

    # optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 14

    use_adam = False  # Use Adam optimizer
    use_chainer = False  # try match everything as chainer
    use_drop = False  # use dropout in RoIHead

    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    load_path = None

    caffe_pretrain = False  # use caffe pretrained model instead of torchvision
    vgg_caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'
    res_caffe_pretrain_path = 'checkpoints/resnet101-caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()