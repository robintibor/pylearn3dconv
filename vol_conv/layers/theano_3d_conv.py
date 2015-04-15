from conv_3d import Conv3dElemwise
import functools
from theano.tensor.nnet.Conv3D import Conv3D
from pylearn2.linear.linear_transform import LinearTransform as P2LT

class Theano3dConv():
    op_axes = ('b', 0, 1, 2, 'c')
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

        rval = Conv3D()(x, self.filters, self.bias, d=tuple(self.kernel_stride))

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


class Theano3dConv3dElemwise(Conv3dElemwise):
    conv_theano_op=Theano3dConv