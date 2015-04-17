from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import sharedX, wraps
from pylearn2.models.mlp import Layer, BadInputSpaceError
import logging
from pylearn3dconv.volumetric_space import Conv3DSpace
import numpy as np
import theano.tensor as T
from theano.compat import OrderedDict
from pylearn3dconv.theanodnn.pool import dnn_pool3d2d

logger = logging.getLogger(__name__)

default_seed = hash('tobipuma') % 4294967295 # good seed is important ;)

def make_conv_3d(irange, input_space, output_space,
        kernel_shape, kernel_stride, conv_op, init_bias=0., rng=None):
    rng = make_np_rng(rng, default_seed, which_method='uniform')
    weights_shape = _get_weights_shape(out_channels = output_space.num_channels,
        kernel_shape=kernel_shape, in_channels=input_space.num_channels, 
        conv_op_axes = conv_op.op_axes)
    
    W = sharedX(rng.uniform(-irange, irange, weights_shape))
    bias = sharedX(np.zeros(weights_shape[0]).astype('float32') + init_bias)
    return conv_op(
        filters=W,
        bias=bias,
        kernel_stride = kernel_stride,
        input_axes=input_space.axes,
        output_axes=output_space.axes
    )

def _get_weights_shape(out_channels, kernel_shape, in_channels, conv_op_axes):
    """ Get shape of weights for given conv_op_axes
    >>> _get_weights_shape(3, [4, 5, 6], 7, ('b', 0, 1, 2, 'c'))
    [3, 4, 5, 6, 7]
    >>> _get_weights_shape(3, [4, 5, 6], 7, ('b', 'c', 0, 1, 2))
    [3, 7, 4, 5, 6]
    """
    # Use b 0 1 2 c shape and then shuffle it according to op axes 
    weights_shape_b_0_1_2_c = [out_channels, kernel_shape[0], kernel_shape[1], 
        kernel_shape[2], in_channels]
    shuffle_arr = [['b', 0, 1, 2, 'c'].index(ax) for ax in conv_op_axes]
    weights_shape = [weights_shape_b_0_1_2_c[i] for i in shuffle_arr]
    return weights_shape

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

    conv_transformer=None # should be overwritten by subclass
    pool_transformer=None # should be overwritten by subclass

    def __init__(self,
                 output_channels,
                 kernel_shape,
                 kernel_stride,
                 layer_name,                 
                 nonlinearity,
                 irange,
                 init_bias=0.,
                 pool_type=None):
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
        self.transformer = make_conv_3d(
            irange=self.irange,
            input_space=self.input_space,
            output_space=self.detector_space,
            kernel_shape=self.kernel_shape,
            kernel_stride=self.kernel_stride,
            conv_op=self.conv_transformer,
            rng=rng)

    def initialize_output_space(self):
        """
        Initializes the output space of the ConvElemwise layer by taking
        pooling operator and the hyperparameters of the convolutional layer
        into consideration as well.
        """
        if self.pool_type is not None:
            dummy_batch_size = self.mlp.batch_size
            if dummy_batch_size is None:
                dummy_batch_size = 2
            dummy_detector =\
                sharedX(self.detector_space.get_origin_batch(dummy_batch_size))
            assert self.pool_type in ['max', 'mean']
            
            # rename pool type for dnn ('mean' should be 'average')
            pool_type = self.pool_type
            if pool_type =='max': pool_type = 'average'
            
            dummy_p = dnn_pool3d2d(inputs=dummy_detector,
                               pool_shape=self.pool_shape,
                               pool_stride=self.pool_stride,
                               image_shape=self.detector_space.shape,
                               mode=pool_type)
            dummy_p = dummy_p.eval()
            self.output_space = Conv3DSpace(shape=[dummy_p.shape[2],
                                                   dummy_p.shape[3],
                                                   dummy_p.shape[4]],
                                            num_channels=self.output_channels,
                                            axes=('b', 'c', 0, 1, 2))
        else:
            # no pooling so set output space to detector space
            self.output_space = self.detector_space
        
        # TODOREMOVETHIS:
        self.output_space = self.detector_space

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
        # output shape determined by input/kernel shapes and stride
        output_shape = [int((self.input_space.shape[i] - 
                            self.kernel_shape[i]) / 
                        self.kernel_stride[i]) + 1 
                        for i in xrange(3)]

        self.detector_space = Conv3DSpace(shape=output_shape,
                                          num_channels=self.output_channels,
                                          axes=self.conv_transformer.op_axes)

        self.initialize_transformer(rng)

        W, bias = self.transformer.get_params()
        W.name = self.layer_name + '_W'
        self.b = bias
        self.b.name = self.layer_name + '_b'
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
        cost = self.cost
        orval = self.nonlin.get_monitoring_channels_from_state(state,
                                                               targets,
                                                               cost_fn=cost)
        rval.update(orval)
        return rval

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        self.input_space.validate(state_below)
        z = self.transformer.lmul(state_below) # will already apply bias
        d = self.nonlin.apply(z)

        if self.layer_name is not None:
            d.name = self.layer_name + '_z'
            self.detector_space.validate(d)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            d = self.output_normalization(d)

        return d