import logging
import numpy as np

from theano.compat import OrderedDict
from theano import tensor as T

from pylearn2.models.mlp import Layer, max_pool, mean_pool, BadInputSpaceError
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX

from pylearn2.linear.conv2d import Conv2D

from pylearn2.expr.nnet import kl

from pylearn2.utils import wraps
from pylearn2.utils.rng import make_np_rng
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.sandbox.cuda.blas import GpuCorrMM

default_seed = [2015, 15, 3, 2]

logger = logging.getLogger(__name__)

class Corr2D(Conv2D):
    def __init__(self, filters, batch_size, input_space,
            output_axes=('b', 0, 1, 'c'), subsample=(1, 1),
            border_mode='valid', filters_shape=None, message=''):
        super(Corr2D, self).__init__(filters, batch_size, input_space,
                output_axes, subsample, border_mode, filters_shape, message)


    def lmul(self, x):
        assert x.ndim == 4
        axes = self.input_space.axes
        assert len(axes) == 4

        op_axes = ('b', 'c', 0, 1)


        if tuple(axes) != op_axes:
            x = x.dimshuffle(
                    axes.index('b'),
                    axes.index('c'),
                    axes.index(0),
                    axes.index(1))

        x = gpu_contiguous(x)

        rval = GpuCorrMM(border_mode=self._border_mode,
                         subsample=self._subsample)(
                             x, self._filters)

        axes = self.output_axes
        assert len(axes) == 4

        if tuple(axes) != op_axes:
            rval = rval.dimshuffle(
                op_axes.index(axes[0]),
                op_axes.index(axes[1]),
                op_axes.index(axes[2]),
                op_axes.index(axes[3])
            )

        return rval

def make_random_conv2D(irange, input_space, output_space,
        kernel_shape, batch_size=None,
        subsample=(1, 1), border_mode='valid', message='', rng=None):
    rng = make_np_rng(rng, default_seed, which_method='uniform')

    W = sharedX(rng.uniform(
        -irange, irange,
        (output_space.num_channels, input_space.num_channels,
         kernel_shape[0], kernel_shape[1])
    ))

    return Corr2D(
        filters=W,
        batch_size=batch_size,
        input_space=input_space,
        output_axes=output_space.axes,
        subsample=subsample, border_mode=border_mode,
        filters_shape=W.get_value(borrow=True).shape, message=message
    )

class ConvElemwiseBlas(Layer):
    """
    Generic convolutional elemwise layer.
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
    pool_shape : tuple
        The shape of the spatial max pooling. A two-tuple of ints.
    pool_stride : tuple
        The stride of the spatial max pooling. Also must be square.
    layer_name : str
        A name for this layer that will be prepended to monitoring channels
        related to this layer.
    nonlinearity : object
        An instance of a nonlinearity object which might be inherited
        from the ConvNonlinearity class.
    irange : float, optional
        if specified, initializes each weight randomly in
        U(-irange, irange)
    border_mode : str, optional
        A string indicating the size of the output:

          - "full" : The output is the full discrete linear convolution of the
            inputs.
          - "valid" : The output consists only of those elements that do not
            rely on the zero-padding. (Default)
    sparse_init : WRITEME
    include_prob : float, optional
        probability of including a weight element in the set of weights
        initialized to U(-irange, irange). If not included it is initialized
        to 1.0.
    init_bias : float, optional
        All biases are initialized to this number. Default is 0.
    W_lr_scale : float or None
        The learning rate on the weights for this layer is multiplied by this
        scaling factor
    b_lr_scale : float or None
        The learning rate on the biases for this layer is multiplied by this
        scaling factor
    max_kernel_norm : float or None
        If specified, each kernel is constrained to have at most this norm.
    pool_type : str or None
        The type of the pooling operation performed the convolution.
        Default pooling type is max-pooling.
    tied_b : bool, optional
        If true, all biases in the same channel are constrained to be the
        same as each other. Otherwise, each bias at each location is
        learned independently. Default is true.
    detector_normalization : callable or None
        See `output_normalization`.
        If pooling argument is not provided, detector_normalization
        is not applied on the layer.
    output_normalization : callable  or None
        if specified, should be a callable object. the state of the
        network is optionally replaced with normalization(state) at each
        of the 3 points in processing:

          - detector: the maxout units can be normalized prior to the
            spatial pooling
          - output: the output of the layer, after sptial pooling, can
            be normalized as well
    kernel_stride : 2-tuple of ints, optional
        The stride of the convolution kernel. Default is (1, 1).
    """

    def __init__(self,
                 output_channels,
                 kernel_shape,
                 layer_name,
                 nonlinearity,
                 irange=None,
                 border_mode='valid',
                 sparse_init=None,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 max_kernel_norm=None,
                 pool_type=None,
                 pool_shape=None,
                 pool_stride=None,
                 tied_b=None,
                 detector_normalization=None,
                 output_normalization=None,
                 kernel_stride=(1, 1),
                 monitor_style="classification"):
        super(ConvElemwiseBlas, self).__init__()

        if (irange is None) and (sparse_init is None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvElemwise.")
        elif (irange is not None) and (sparse_init is not None):
            raise AssertionError("You should specify either irange or "
                                 "sparse_init when calling the constructor of "
                                 "ConvElemwise and not both.")

        if pool_type is not None:
            assert pool_shape is not None, (
                "You should specify the shape of "
                "the spatial %s-pooling." % pool_type)
            assert pool_stride is not None, (
                "You should specify the strides of "
                "the spatial %s-pooling." % pool_type)

        assert nonlinearity is not None

        self.nonlin = nonlinearity
        self.__dict__.update(locals())
        assert monitor_style in ['classification', 'detection'], (
            "%s.monitor_style should be either"
            "detection or classification" % self.__class__.__name__)
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
        if self.irange is not None:
            assert self.sparse_init is None
            self.transformer = make_random_conv2D(
                irange=self.irange,
                input_space=self.input_space,
                output_space=self.detector_space,
                kernel_shape=self.kernel_shape,
                subsample=self.kernel_stride,
                border_mode=self.border_mode,
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

        if self.pool_type is not None:
            assert self.pool_type in ['max', 'mean']
            if self.pool_type == 'max':
                dummy_p = max_pool(bc01=dummy_detector,
                                   pool_shape=self.pool_shape,
                                   pool_stride=self.pool_stride,
                                   image_shape=self.detector_space.shape)
            elif self.pool_type == 'mean':
                dummy_p = mean_pool(bc01=dummy_detector,
                                    pool_shape=self.pool_shape,
                                    pool_stride=self.pool_stride,
                                    image_shape=self.detector_space.shape)
            dummy_p = dummy_p.eval()
            self.output_space = Conv2DSpace(shape=[dummy_p.shape[2],
                                                   dummy_p.shape[3]],
                                            num_channels=self.output_channels,
                                            axes=('b', 'c', 0, 1))
        else:
            dummy_detector = dummy_detector.eval()
            self.output_space = Conv2DSpace(shape=[dummy_detector.shape[2],
                                            dummy_detector.shape[3]],
                                            num_channels=self.output_channels,
                                            axes=('b', 'c', 0, 1))

        logger.info('Output space: {0}'.format(self.output_space.shape))

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        """ Note: this function will reset the parameters! """

        self.input_space = space

        if not isinstance(space, Conv2DSpace):
            raise BadInputSpaceError(self.__class__.__name__ +
                                     ".set_input_space "
                                     "expected a Conv2DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))

        rng = self.mlp.rng

        if self.border_mode == 'valid':
            output_shape = [int((self.input_space.shape[0]
                                 - self.kernel_shape[0])
                                / self.kernel_stride[0]) + 1,
                            int((self.input_space.shape[1]
                                 - self.kernel_shape[1])
                                / self.kernel_stride[1]) + 1]
        elif self.border_mode == 'full':
            output_shape = [int((self.input_space.shape[0]
                                 + self.kernel_shape[0])
                                / self.kernel_stride[0]) - 1,
                            int((self.input_space.shape[1]
                                 + self.kernel_shape[1])
                                / self.kernel_stride[1]) - 1]

        self.detector_space = Conv2DSpace(shape=output_shape,
                                          num_channels=self.output_channels,
                                          axes=('b', 'c', 0, 1))

        self.initialize_transformer(rng)

        W, = self.transformer.get_params()
        W.name = self.layer_name + '_W'

        if self.tied_b:
            self.b = sharedX(np.zeros((self.detector_space.num_channels)) +
                             self.init_bias)
        else:
            self.b = sharedX(self.detector_space.get_origin() + self.init_bias)

        self.b.name = self.layer_name + '_b'

        logger.info('Input shape: {0}'.format(self.input_space.shape))
        logger.info('Detector space: {0}'.format(self.detector_space.shape))

        self.initialize_output_space()

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):
        if self.max_kernel_norm is not None:
            W, = self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(1, 2, 3)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                updates[W] = updated_W * (
                    desired_norms /
                    (1e-7 + row_norms)).dimshuffle(0, 'x', 'x', 'x')

    @wraps(Layer.get_params)
    def get_params(self):
        assert self.b.name is not None
        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

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

        W, = self.transformer.get_params()
        W.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):

        self.b.set_value(biases)

    @wraps(Layer.get_biases)
    def get_biases(self):

        return self.b.get_value()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):

        return ('v', 'h')

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

        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw, (outp, rows, cols, inp))

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):

        W, = self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(1, 2, 3)))

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
        if not hasattr(self, 'tied_b'):
            self.tied_b = False

        if self.tied_b:
            b = self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            b = self.b.dimshuffle('x', 0, 1, 2)

        z = z + b
        d = self.nonlin.apply(z)

        if self.layer_name is not None:
            d.name = self.layer_name + '_z'
            self.detector_space.validate(d)

        if self.pool_type is not None:
            if not hasattr(self, 'detector_normalization'):
                self.detector_normalization = None

            if self.detector_normalization:
                d = self.detector_normalization(d)

            assert self.pool_type in ['max', 'mean'], ("pool_type should be"
                                                       "either max or mean"
                                                       "pooling.")

            if self.pool_type == 'max':
                p = max_pool(bc01=d, pool_shape=self.pool_shape,
                             pool_stride=self.pool_stride,
                             image_shape=self.detector_space.shape)
            elif self.pool_type == 'mean':
                p = mean_pool(bc01=d, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)

            self.output_space.validate(p)
        else:
            p = d

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p

    def cost(self, Y, Y_hat):
        """
        Cost for convnets is hardcoded to be the cost for sigmoids.
        TODO: move the cost into the non-linearity class.

        Parameters
        ----------
        Y : theano.gof.Variable
            Output of `fprop`
        Y_hat : theano.gof.Variable
            Targets

        Returns
        -------
        cost : theano.gof.Variable
            0-D tensor describing the cost

        Notes
        -----
        Cost mean across units, mean across batch of KL divergence
        KL(P || Q) where P is defined by Y and Q is defined by Y_hat
        KL(P || Q) = p log p - p log q + (1-p) log (1-p) - (1-p) log (1-q)
        """
        assert self.nonlin.non_lin_name == "sigmoid", ("ConvElemwise "
                                                       "supports "
                                                       "cost function "
                                                       "for only "
                                                       "sigmoid layer "
                                                       "for now.")
        batch_axis = self.output_space.get_batch_axis()
        ave_total = kl(Y=Y, Y_hat=Y_hat, batch_axis=batch_axis)
        ave = ave_total.mean()
        return ave

