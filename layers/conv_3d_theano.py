import functools
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import sharedX, wraps
from theano.tensor.nnet.Conv3D import Conv3D
from pylearn2.models.mlp import Layer, BadInputSpaceError
import logging
from volumetric_space import Conv3DSpace
import numpy as np
import theano.tensor as T
from theano.compat import OrderedDict

logger = logging.getLogger(__name__)
from pylearn2.linear.linear_transform import LinearTransform as P2LT

default_seed = 42764187 # some random number :)

class Theano3dConv():
    def __init__(self, filters, bias, input_space, output_axes):
        self.__dict__.update(locals())

    def lmul(self, x):
        assert x.ndim == 5
        input_axes = self.input_space.axes
        assert len(input_axes) == 5

        op_axes = ('b', 0, 1,  2, 'c')

        if tuple(input_axes) != op_axes:
            # convert from input axes to op_axes
            reshuffle_arr = [input_axes.index(op_axes[i]) for i in xrange(5)]
            x = x.dimshuffle(*reshuffle_arr)

        #for now fakebias
        #bias = T.fvector()
        #bias = T.zeros_like(self.filters[:, 0,0,0,0]) 
        rval = Conv3D()(x, self.filters, self.bias, d=(1,1,1))

        output_axes = self.output_axes
        assert len(output_axes) == 5

        if tuple(output_axes) != op_axes:
            # convert from op axes to output axes
            reshuffle_arr = [op_axes.index(output_axes[i]) for i in xrange(5)]
            rval = rval.dimshuffle(*reshuffle_arr)

        return rval

    @functools.wraps(P2LT.get_params)
    def get_params(self):
        """
        .. todo::
            WRITEME
        """
        return [self.filters, self.bias]

def make_theano_conv_3d(irange, input_space, output_space,
        kernel_shape, init_bias=0., rng=None):
    rng = make_np_rng(rng, default_seed, which_method='uniform')

    # needs to correspond to expected shape of filters for 3d conv
    # (out channel, row, column, time ,in channel) according to theano doc
    weights_shape = (output_space.num_channels, kernel_shape[0], kernel_shape[1],
            kernel_shape[2], input_space.num_channels)
    W = sharedX(rng.uniform(-irange, irange, weights_shape))
    bias = sharedX(np.zeros(weights_shape[0]).astype('float32') + init_bias)
    return Theano3dConv(
        filters=W,
        bias=bias,
        input_space=input_space,
        output_axes=output_space.axes,
    )





class Conv3dElemwise(Layer):
    """
    Generic convolutional 3d elemwise layer.
    Takes the ConvNonlinearity object as an argument and implements
    convolutional layer with the specified nonlinearity.
    This function can implement:
    * Linear convolutional layer
    * Rectifier convolutional layer
    * Sigmoid convolutional layer
    * Tanh convolutional layer
    based on the nonlinearity argument that it recieves.
    Parameters
    ----------
    output_channels : int
        The number of output channels the layer should have.
    kernel_shape : tuple
        The shape of the convolution kernel.
    layer_name : str
        A name for this layer that will be prepended to monitoring channels
        related to this layer.
    nonlinearity : object
        An instance of a nonlinearity object which might be inherited
        from the ConvNonlinearity class.
    irange : float
        Initializes each weight randomly in
        U(-irange, irange)
    
    sparse_init : WRITEME
    
    """

    def __init__(self,
                 output_channels,
                 kernel_shape,
                 layer_name,
                 nonlinearity,
                 irange,
                 init_bias=0.):
        super(Conv3dElemwise, self).__init__()
        assert nonlinearity is not None

        self.nonlin = nonlinearity
        self.__dict__.update(locals())
        del self.self

    def initialize_transformer(self, rng):
        """
        This function initializes the transformer of the class. Re-running
        this function will reset the transformer.
        Parameters
        ----------
        rng : object
            random number generator object.
        """
        assert self.irange is not None
        self.transformer = make_theano_conv_3d(
            irange=self.irange,
            input_space=self.input_space,
            output_space=self.detector_space,
            kernel_shape=self.kernel_shape,
            rng=rng)

    def initialize_output_space(self):
        """
        Initializes the output space of the ConvElemwise layer by taking
        pooling operator and the hyperparameters of the convolutional layer
        into consideration as well.
        """
        dummy_batch_size = self.mlp.batch_size

        if dummy_batch_size is None:
            dummy_batch_size = 2
        dummy_detector =\
            sharedX(self.detector_space.get_origin_batch(dummy_batch_size))

        dummy_detector = dummy_detector.eval()
        self.output_space = Conv3DSpace(shape=dummy_detector.shape[1:4],
                                        num_channels=self.output_channels,
                                        axes=self.detector_space.axes)

        logger.info('Output space: {0}'.format(self.output_space.shape))

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        """ Note: this function will reset the parameters! """

        self.input_space = space

        if not isinstance(space, Conv3DSpace):
            raise BadInputSpaceError(self.__class__.__name__ +
                                     ".set_input_space "
                                     "expected a Conv3DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))

        rng = self.mlp.rng

        output_shape = [int(self.input_space.shape[0]
                             - self.kernel_shape[0]) + 1,
                        int(self.input_space.shape[1]
                             - self.kernel_shape[1]) + 1,
                        int(self.input_space.shape[2]
                             - self.kernel_shape[2]) + 1]
        

        self.detector_space = Conv3DSpace(shape=output_shape,
                                          num_channels=self.output_channels,
                                          axes=('b',  0, 1, 2, 'c'))

        self.initialize_transformer(rng)

        W, bias = self.transformer.get_params()
        W.name = self.layer_name + '_W'
        self.b = bias
        self.b.name = self.layer_name + '_b'
        """
        if self.tied_b:
            
        else:
            self.b = sharedX(self.detector_space.get_origin() + self.init_bias)
"""

        logger.info('Input shape: {0}'.format(self.input_space.shape))
        logger.info('Detector space: {0}'.format(self.detector_space.shape))

        self.initialize_output_space()

    @wraps(Layer.get_params)
    def get_params(self):
        W,b = self.transformer.get_params()
        assert W.name is not None
        assert b.name is not None
        return [W,b]

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * abs(W).sum()

    @wraps(Layer.set_weights)
    def set_weights(self, weights):
        W, _ = self.transformer.get_params()
        W.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):
        self.b.set_value(biases)

    @wraps(Layer.get_biases)
    def get_biases(self):
        return self.b.get_value()

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):
        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):
        return self.transformer._filters.get_value()

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):

        W, = self.transformer.get_params()

        assert W.ndim == 5

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(1, 2, 3, 4)))

        rval = OrderedDict([
                           ('kernel_norms_min', row_norms.min()),
                           ('kernel_norms_mean', row_norms.mean()),
                           ('kernel_norms_max', row_norms.max()),
                           ])

        cst = self.cost
        orval = self.nonlin.get_monitoring_channels_from_state(state,
                                                               targets,
                                                               cost_fn=cst)

        rval.update(orval)

        return rval

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        self.input_space.validate(state_below)

        z = self.transformer.lmul(state_below)
        """ ignore bias for now
        if not hasattr(self, 'tied_b'):
            self.tied_b = False

        if self.tied_b:
            b = self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            b = self.b.dimshuffle('x', 0, 1, 2)

        z = z + b
        """
        d = self.nonlin.apply(z)

        if self.layer_name is not None:
            d.name = self.layer_name + '_z'
            self.detector_space.validate(d)

        p = d

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p
