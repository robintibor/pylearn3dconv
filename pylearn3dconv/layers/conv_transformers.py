from pylearn2.linear.linear_transform import LinearTransform as P2LT
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda.blas import GpuCorr3dMM
from theano.tensor.nnet.Conv3D import Conv3D
from pylearn3dconv.theanodnn.conv import dnn_3dconv
import theano.tensor.nnet.conv3d2d
import functools

class Conv3dTransformer():
    """ Transforms from input to convolved input and adds bias.
    Base class for actual implementations.
    Subclasses need to implement conv_and_add_bias. """
    op_axes = None # should be overwritten by subclass
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
        rval = self.conv_and_add_bias(x)

        output_axes = self.output_axes
        assert len(output_axes) == 5

        if tuple(output_axes) != self.op_axes:
            # convert from op axes to output axes
            reshuffle_arr = [self.op_axes.index(output_axes[i]) for i in xrange(5)]
            rval = rval.dimshuffle(*reshuffle_arr)
        return rval
    
    def conv_and_add_bias(self, x):
        raise NotImplementedError("This class should not be used directly.\n"
            "Use one of the subclasses.")

    @functools.wraps(P2LT.get_params)
    def get_params(self):
        """
        Returns filters and biases.
        Returns
        -------
        filters : theano shared variable
        bias : theano shared variable
        """
        return [self.filters, self.bias]


class CuDnn3dConv(Conv3dTransformer):
    op_axes = ('b', 'c', 0, 1, 2)
    def conv_and_add_bias(self, x):
        rval = dnn_3dconv(x, self.filters, subsample=self.kernel_stride,
            conv_mode='cross')
        rval = rval + self.bias.dimshuffle('x', 0, 'x', 'x', 'x')
        return rval

class CuBlas3dConv(Conv3dTransformer):
    op_axes = ('b', 'c', 0, 1, 2)
    def conv_and_add_bias(self, x):
        x  = gpu_contiguous(x)
        rval = GpuCorr3dMM(subsample=tuple(self.kernel_stride))(x, self.filters)
        rval = rval + self.bias.dimshuffle('x', 0, 'x', 'x', 'x')
        return rval
    
class Theano3dConv(Conv3dTransformer):
    op_axes = ('b', 0, 1, 2, 'c')
    def conv_and_add_bias(self, x):
        rval = Conv3D()(x, self.filters, self.bias, d=tuple(self.kernel_stride))
        return rval

class Theano3d2dConv(Conv3dTransformer):
    op_axes = ('b', 2, 'c', 0, 1)
    def conv_and_add_bias(self, x):
        rval = theano.tensor.nnet.conv3d2d.conv3d(x, self.filters)
        rval = self._subsample(rval, tuple(self.kernel_stride))
        rval = rval + self.bias.dimshuffle('x', 'x', 0, 'x', 'x')
        return rval
    
    def _subsample(self, result, kernel_stride):
        """ Subsample result with stride if necessary.
        Expects kernel_stride to be a tuple."""
        assert isinstance(kernel_stride, tuple)
        if (kernel_stride != (1,1,1)):
            result = result[:,::kernel_stride[2], :, ::kernel_stride[0],
                ::kernel_stride[1]]
        return result
