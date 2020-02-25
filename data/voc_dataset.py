import os
import numpy as np
import xml.etree.ElementTree as ET
from data.util import read_image

class VOCBboxDataset:
    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,
                 ):
        # 读取id列表
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))
        self.data_dir = data_dir
        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, idx):
        # 返回对应id的RGB图片（CHW format）及bounding boxes
        id_ = self.ids[idx]
        # 使用python标准库xml.etree.ElementTree来访问xml文件
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        # 读取obj标注信息存入列表
        for obj in anno.findall('object'):
            # 在not using difficult的split中跳过difficult object
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # 找到bounding box坐标并-1使其从0开始
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)

        # 根据地址得到RGB图片
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        return img, bbox, label, difficult

    __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
)