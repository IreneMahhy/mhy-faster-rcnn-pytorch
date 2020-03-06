from collections import namedtuple
from string import Template

import cupy as cp
import torch as t
from torch import nn

from torch.autograd import Function
from model.utils.roi_cupy import kernel_backward, kernel_forward

Stream = namedtuple('Stream', ['ptr'])


@cp.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    cp.cuda.runtime.free(0)
    code = Template(code).substitute(**kwargs)
    kernel_code = cp.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K


class RoI(Function):
    @staticmethod
    def forward(ctx, x, rois, outh, outw, spatial_scale):
        forward_fn = load_kernel('roi_forward', kernel_forward)
        backward_fn = load_kernel('roi_backward', kernel_backward)
        x = x.contiguous()
        rois = rois.contiguous()
        in_size = B, C, H, W = x.size()
        N = rois.size(0)   # RoI的总数
        output = t.zeros(N, C, outh, outw).cuda()  # pooling的输出
        argmax_data = t.zeros(N, C, outh, outw).int().cuda()
        args = [x.data_ptr(), rois.data_ptr(),
                output.data_ptr(), argmax_data.data_ptr(),
                spatial_scale, C, H, W, outh,
                outw, output.numel()]  # numel()返回元素个数
        stream = Stream(ptr=t.cuda.current_stream().cuda_stream)
        forward_fn(args=args,
                   block=(CUDA_NUM_THREADS, 1, 1),
                   grid=(GET_BLOCKS(output.numel()), 1, 1),
                   stream=stream)
        ctx.N = N
        ctx.outh = outh
        ctx.outw = outw
        ctx.spatial_scale = spatial_scale
        ctx.in_size = in_size
        ctx.argmax_data = argmax_data
        ctx.rois = rois
        ctx.forward_fn = forward_fn
        ctx.backward_fn = backward_fn

        return output

    @staticmethod
    def backward(ctx, grad_output):
        N = ctx.N
        outh = ctx.outh
        outw = ctx.outw
        spatial_scale = ctx.spatial_scale
        in_size = ctx.in_size
        argmax_data = ctx.argmax_data
        rois = ctx.rois
        forward_fn = ctx.forward_fn
        backward_fn = ctx.backward_fn
        grad_output = grad_output.contiguous()
        B, C, H, W = in_size
        grad_input = t.zeros(in_size).cuda()
        stream = Stream(ptr=t.cuda.current_stream().cuda_stream)
        args = [grad_output.data_ptr(),
                argmax_data.data_ptr(),
                rois.data_ptr(),
                grad_input.data_ptr(),
                N, spatial_scale, C, H, W, outh, outw,
                grad_input.numel()]
        backward_fn(args=args,
                    block=(CUDA_NUM_THREADS, 1, 1),
                    grid=(GET_BLOCKS(grad_input.numel()), 1, 1),
                    stream=stream)

        return grad_input, None, None, None, None


class RoIPooling2D(nn.Module):
    def __init__(self, outh, outw, spatial_scale):
        super(RoIPooling2D, self).__init__()
        self.outh = outh
        self.outw = outw
        self.spatial_scale = spatial_scale

    def forward(self, x, rois):
        return RoI.apply(x, rois, self.outh, self.outw, self.spatial_scale)


def test_roi_module():
    ## fake data###
    B, N, C, H, W, PH, PW = 2, 8, 4, 32, 32, 7, 7

    bottom_data = t.randn(B, C, H, W).cuda()
    bottom_rois = t.randn(N, 5)
    bottom_rois[:int(N / 2), 0] = 0
    bottom_rois[int(N / 2):, 0] = 1
    bottom_rois[:, 1:] = (t.rand(N, 4) * 100).float()
    bottom_rois = bottom_rois.cuda()
    spatial_scale = 1. / 16
    outh, outw = PH, PW

    # pytorch version
    module = RoIPooling2D(outh, outw, spatial_scale)
    x = bottom_data.requires_grad_()
    rois = bottom_rois.detach()

    output = module(x, rois)
    output.sum().backward()

    def t2c(variable):
        npa = variable.data.cpu().numpy()
        return cp.array(npa)

    def test_eq(variable, array, info):
        cc = cp.asnumpy(array)
        neq = (cc != variable.data.cpu().numpy())
        assert neq.sum() == 0, 'test failed: %s' % info

    # chainer version,if you're going to run this
    # pip install chainer
    import chainer.functions as F
    from chainer import Variable
    x_cn = Variable(t2c(x))

    o_cn = F.roi_pooling_2d(x_cn, t2c(rois), outh, outw, spatial_scale)
    test_eq(output, o_cn.array, 'forward')
    F.sum(o_cn).backward()
    test_eq(x.grad, x_cn.grad, 'backward')
    print('test pass')