from conv_3d import Conv3dElemwise
import functools
from pylearn2.linear.linear_transform import LinearTransform as P2LT
from theano.sandbox.cuda.basic_ops import (gpu_contiguous, gpu_alloc_empty)
#from pylearn3dconv.theanodnn3d.theano_dnn_conv import GpuDnn3dConv, GpuDnnConv3dDesc
from pylearn3dconv.theanodnn3d.theano_dnn_conv import dnn_3dconv

class CuDnn3dConv():
    op_axes = ('b', 'c', 0, 1, 2)
    def __init__(self, filters, bias, kernel_stride, input_axes, output_axes):
        self.__dict__.update(locals())
        del self.self

    def lmul(self, x):
        assert x.ndim == 5
        input_axes = self.input_axes
        assert len(input_axes) == 5
        if tuple(input_axes) != self.op_axes:
            # convert from input axes to op_axes
            reshuffle_arr = [input_axes.index(self.op_axes[i]) for i in xrange(5)]
            x = x.dimshuffle(*reshuffle_arr)
        rval = dnn_3dconv(x, self.filters, subsample=self.kernel_stride,
            conv_mode='cross')
        rval = rval + self.bias.dimshuffle('x', 0, 'x', 'x', 'x')

        output_axes = self.output_axes
        assert len(output_axes) == 5

        if tuple(output_axes) != self.op_axes:
            # convert from op axes to output axes
            reshuffle_arr = [self.op_axes.index(output_axes[i]) for i in xrange(5)]
            rval = rval.dimshuffle(*reshuffle_arr)
        return rval

    @functools.wraps(P2LT.get_params)
    def get_params(self):
        """
        .. todo::
            WRITEME
        """
        return [self.filters, self.bias]


class CuDnnConv3dElemwise(Conv3dElemwise):
    conv_theano_op=CuDnn3dConv