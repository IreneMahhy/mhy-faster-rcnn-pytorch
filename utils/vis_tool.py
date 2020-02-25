import time
import numpy as np
import torch as t
import matplotlib
import visdom
from data.voc_dataset import VOC_BBOX_LABEL_NAMES

matplotlib.use('Agg')
from matplotlib import pyplot as plot


# 显示一张三通道彩色图片
def vis_image(img, ax=None):
    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)   # 画布只显示一幅图
    img = img.transpose((1, 2, 0))  # CHW -> HWC
    ax.imshow(img.astype(np.uint8))

    return ax


def vis_bbox(img, bbox, label=None, score=None, ax=None):
    # 加入背景类
    label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    ax = vis_image(img, ax=ax)
    if len(bbox) == 0:
        return ax

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])     # bbox左下角点的xy坐标
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plot.Rectangle(xy, width,
                                    height, fill=False,
                                    edgecolor='red', linewidth=2))
        caption = list()
        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 < lb < len(label_names)):
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0], ':'.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})

        return ax


# 将matplotlib图片转换为RGBA三维图片array
def fig2data(fig):
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    buf = np.roll(buf, 3, axis=2)

    return buf.reshape(h, w, 4)


# 封装fig2data，转换为BGR
def fig4vis(fig):
    """
    convert figure to ndarray
    """
    ax = fig.get_figure()
    img_data = fig2data(ax).astype(np.int32)
    plot.close()
    # HWC->CHW
    return img_data[:, :, :3].transpose((2, 0, 1)) / 255.


def visdom_bbox(*args, **kwargs):
    fig = vis_bbox(*args, **kwargs)
    data = fig4vis(fig)

    return data


class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        self._vis_kw = kwargs

        self.index = {}
        self.log_text = ''

    # 改变visdom参数并启动
    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        return self

    def plot_many(self, d):
        for k, v in d.items():
            if v is not None:
                self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        !!don't ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~!!
        """
        self.vis.images(t.Tensor(img_).cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), \
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

    def state_dict(self):
        return {
            'index': self.index,
            'vis_kw': self._vis_kw,
            'log_text': self.log_text,
            'env': self.vis.env
        }

    def load_state_dict(self, d):
        self.vis = visdom.Visdom(env=d.get('env', self.vis.env), **(self.d.get('vis_kw')))
        self.log_text = d.get('log_text', '')
        self.index = d.get('index', dict())
        return self
