# mhy-faster-rcnn-pytorch
This project is a imitative version of [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch) by chenyuntc.  
这是在陈云的simple-faster-rcnn-pytorch基础上参考仿写的对Faster R-CNN的pytorch实现。  


# Modification
1. modified RoIPooling part in model/roi_module.py, changed the forward and backward method into static method (the non-static method is deprecated).  
2. added res101 backbone (the original version only uses vgg16). Caffe-pretrained (on ImageNet) res101 model is needed.
3. pretrained model (.pth) path is needed in utils/config.py.
4. For instructions please follow the ORIGINAL_README.md.
