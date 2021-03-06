import os
import numpy

import theano
from theano import Apply, gof, tensor, config, Variable
from theano.scalar import as_scalar, constant
from theano.gradient import DisconnectedType, grad_not_implemented
from theano.gof import Optimizer, local_optimizer, COp
from theano.gof.type import CDataType, Generic
from theano.compile import optdb
from theano.compile.ops import shape_i
from theano.configparser import AddConfigVar, EnumStr
from theano.tensor.nnet import SoftmaxGrad
from theano.tensor.signal.downsample import (
    DownsampleFactorMax, DownsampleFactorMaxGrad)
from theano.sandbox.cuda import GpuOp
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.blas import (GpuConv, GpuDownsampleFactorMax,
                                      GpuDownsampleFactorMaxGrad)
from theano.sandbox.cuda.nnet import GpuSoftmax
from theano.sandbox.cuda.opt_util import alpha_merge, output_merge
from theano.sandbox.cuda import gpu_seqopt, register_opt

from theano.sandbox.cuda.nvcc_compiler import NVCC_compiler
from theano.sandbox.cuda.dnn import dnn_available, version, DnnBase, \
    DnnVersion, ensure_float, _one, _zero

def dnn_3dconv(img, kerns,  subsample=(1, 1),
             conv_mode='conv'):
    """
    GPU 3d convolution using cuDNN from NVIDIA.

    The memory layout to use is 'bc012', that is 'batch', 'channel',
    'first dim', 'second dim', 'third dim' in that order.

    :param img: images to do the convolution over
    :param kerns: convolution filters
    :param subsample: perform subsampling of the output (default: (1, 1))


    :warning: The cuDNN library only works with GPU that have a compute
      capability of 3.0 or higer.  This means that older GPU will not
      work with this Op.
    """
    img = gpu_contiguous(img)
    kerns = gpu_contiguous(kerns)
    desc = GpuDnnConv3dDesc(subsample=tuple(subsample), 
            conv_mode=conv_mode)()
    desc_op = desc.owner.op
    out_shp = GpuDnn3dConv.get_out_shape(img.shape, kerns.shape,
                                           desc_op.subsample)
        
        
    out = gpu_alloc_empty(*out_shp)
    return GpuDnn3dConv()(img, kerns, out, desc)

class GpuDnnConv3dDesc(GpuOp):
    """This Op builds a 3d convolution descriptor for use in the other
    convolution operations.

    see the doc of :func:`dnn_conv` for a description of the parameters

    """
    __props__ = ('subsample', 'conv_mode')
    
    def c_headers(self):
        return ['cudnn.h']

    def c_libraries(self):
        return ['cudnn']

    def c_compiler(self):
        return NVCC_compiler

    def __init__(self, subsample=(1, 1, 1), conv_mode='conv'):
        #TODELAY: padding
        assert len(subsample) == 3
        self.subsample = subsample
        assert conv_mode in ('conv', 'cross')
        self.conv_mode = conv_mode

    def make_node(self):
        return Apply(self, [],
                     [CDataType("cudnnConvolutionDescriptor_t")()])

    def c_code(self, node, name, inputs, outputs, sub):
        desc, = outputs
        num_dims = 5
        
        # TODELAY pad arr/bordermode
        if self.conv_mode == 'conv':
            conv_flag = 'CUDNN_CONVOLUTION'
        else:
            conv_flag = 'CUDNN_CROSS_CORRELATION'

        return """
{
  cudnnStatus_t err;
  int convDims = 3;
  int padA[3] = {0,0,0}; // TODELAY: no padding for now, could reenable that
  int filterStride[3] = {%(filter_stride)s};
  int upscaleA[3]  = { 1,1,1 }; // don't upscale

  if ((err = cudnnCreateConvolutionDescriptor(&%(desc)s)) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate convolution "
                 "descriptor: %%s", cudnnGetErrorString(err));
    %(fail)s
  }
  // TODELAY: just remove if else, check cudnn version somewhere else
#if defined(CUDNN_VERSION) && CUDNN_VERSION >= 20
  err = cudnnSetConvolutionNdDescriptor(
  %(desc)s,
  convDims, // PyArray_GETPTR1(
  padA,
  filterStride,
  upscaleA,
  %(conv_flag)s
  );
#else
    PyErr_Format(PyExc_RuntimeError, "Need higher dnn version: %%d",
                 CUDNN_VERSION);
  %(fail)s
  );
#endif
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: %%s",
                 cudnnGetErrorString(err));
    %(fail)s
  }
}
""" % dict(num_dims=num_dims, desc=desc,
        conv_flag=conv_flag, fail=sub['fail'],
        # transform subsample tuple to x,y,z string like 1,1,1
        filter_stride = str(tuple(self.subsample)).strip('()'))
    
    def c_code_cache_version(self):
        return (2, version())

class GpuDnn3dConv(DnnBase, COp):
    """
    The forward convolution.

    :param image:
    :param kernel:
    :param descr: the convolution descriptor
    """
    __props__ = ()

    def __init__(self,  inplace=False):
        # TODELAY: reenable inplace optimizations http://deeplearning.net/software/theano/extending/inplace.html
        theano_cuda_dir = os.path.dirname(theano.sandbox.cuda.__file__)
        theano_files = ["dnn_base.c",
            os.path.join(theano_cuda_dir,"dnn_conv_base.c"),
            "dnn_fwd.c"]
        COp.__init__(self, theano_files,
                     "APPLY_SPECIFIC(conv_fwd)")
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [2]}

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'inplace'):
            self.inplace = False

    def get_op_params(self):
        if self.inplace:
            inpl_def = [('CONV_INPLACE', '1')]
        else:
            inpl_def = []
        if version() == -1:
            alg_def = ('CONV_ALGO', "0")
        else:
            # it seems only this works for nd convolutions?
            alg_def = ('CONV_ALGO', 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM')
        return [alg_def] + inpl_def

    def make_node(self, img, kern, output, desc, alpha=None, beta=None):
        img = as_cuda_ndarray_variable(img)
        kern = as_cuda_ndarray_variable(kern)
        output = as_cuda_ndarray_variable(output)
        if img.type.ndim != 5:
            raise TypeError('img must be 5D tensor')
        if kern.type.ndim != 5:
            raise TypeError('kern must be 5D tensor')
        if output.type.ndim != 5:
            raise TypeError('output must be a 5D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        alpha = ensure_float(alpha, _one, 'alpha')
        beta = ensure_float(beta, _zero, 'beta')

        return Apply(self, [img, kern, output, desc, alpha, beta],
                     [output.type()])

    def grad(self, inp, grads):
        img, kerns, output, desc, alpha, beta = inp
        top, = grads

        top = gpu_contiguous(top)

        d_img = GpuDnn3dConvGradI()(kerns, top, gpu_alloc_empty(*img.shape), desc)
        d_kerns = GpuDnn3dConvGradW()(img, top, gpu_alloc_empty(*kerns.shape), desc)
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)

        return [d_img * alpha, d_kerns * alpha, top * beta,
                DisconnectedType()(), d_alpha, d_beta]

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [1], [0], [1], [1]]
    
    
    @staticmethod
    def get_out_shape(ishape, kshape, subsample):
        """
        This function computes the output shape for a convolution with
        the specified parameters.  `ishape` and `kshape` can be symbolic
        or scalar.
        """
        b = ishape[0]  # Number of inputs
        h = ishape[2]  # Height of input feature maps
        w = ishape[3]  # Width of input feature maps
        d = ishape[4]  # Depth of input feature maps
        nb = kshape[0]  # Number of output feature maps
        kh = kshape[2]  # Height of each filter
        kw = kshape[3]  # Width of each filter
        kd = kshape[4]  # Depth of each filter

        sh, sw, sd = subsample

        return (
            b, nb,
            (h - kh)//sh + 1,
            (w - kw)//sw + 1,
            (d - kd)//sd + 1,
        )

    def infer_shape(self, node, shape):
        return [shape[2]]

class GpuDnn3dConvGradW(DnnBase, COp):
    """
    The convolution gradient with respect to the weights.

    :param image:
    :param kernel:
    :param descr: the convolution descriptor

    """
    __props__ = ('inplace',)

    def __init__(self, inplace=False):
        theano_cuda_dir = os.path.dirname(theano.sandbox.cuda.__file__)
        COp.__init__(self, ["dnn_base.c", 
                        os.path.join(theano_cuda_dir,"dnn_conv_base.c"),
                        "dnn_gw.c"],
                     "APPLY_SPECIFIC(conv_gw)")
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [2]}

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'inplace'):
            self.inplace = False

    def grad(self, inp, grads):
        img, top, output, desc, alpha, beta = inp
        kerns, = grads

        kerns = gpu_contiguous(kerns)

        d_img = GpuDnn3dConvGradI()(kerns, top, gpu_alloc_empty(*img.shape), desc)
        d_top = GpuDnn3dConv()(img, kerns, gpu_alloc_empty(*top.shape), desc)
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)

        return (d_img * alpha, d_top * alpha, kerns * beta,
                DisconnectedType()(), d_alpha, d_beta)

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [1], [0], [1], [1]]

    def get_op_params(self):
        if self.inplace:
            return [('CONV_INPLACE', '1')]
        else:
            return []

    def make_node(self, img, topgrad, output, desc, alpha=None, beta=None):
        img = as_cuda_ndarray_variable(img)
        topgrad = as_cuda_ndarray_variable(topgrad)
        output = as_cuda_ndarray_variable(output)
        if img.type.ndim != 5:
            raise TypeError('img must be 5D tensor')
        if topgrad.type.ndim != 5:
            raise TypeError('topgrad must be 5D tensor')
        if output.type.ndim != 5:
            raise TypeError('output must be 5D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        alpha = ensure_float(alpha, _one, 'alpha')
        beta = ensure_float(beta, _zero, 'beta')

        return Apply(self, [img, topgrad, output, desc, alpha, beta],
                     [output.type()])

    def infer_shape(self, node, shape):
        return [shape[2]]

class GpuDnn3dConvGradI(DnnBase, COp):
    """
    The convolution gradient with respect to the inputs.

    :param image:
    :param kernel:
    :param descr: the convolution descriptor

    """
    __props__ = ('inplace',)

    def __init__(self, inplace=False):
        theano_cuda_dir = os.path.dirname(theano.sandbox.cuda.__file__)
        COp.__init__(self, ["dnn_base.c", 
                        os.path.join(theano_cuda_dir,"dnn_conv_base.c"),
                        "dnn_gi.c"],
                     "APPLY_SPECIFIC(conv_gi)")
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [2]}

    def grad(self, inp, grads):
        kerns, top, output, desc, alpha, beta = inp
        img, = grads

        img = gpu_contiguous(img)

        d_kerns = GpuDnn3dConvGradW()(img, top, gpu_alloc_empty(*kerns.shape), desc)
        d_top = GpuDnn3dConv()(img, kerns, gpu_alloc_empty(*top.shape), desc)
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)
        return (d_kerns * alpha, d_top * alpha, img * beta,
                DisconnectedType()(), d_alpha, d_beta)

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [1], [0], [1], [1]]

    def get_op_params(self):
        if self.inplace:
            return [('CONV_INPLACE', '1')]
        else:
            return []

    def make_node(self, kern, topgrad, output, desc, alpha=None, beta=None):
        kern = as_cuda_ndarray_variable(kern)
        topgrad = as_cuda_ndarray_variable(topgrad)
        output = as_cuda_ndarray_variable(output)
        if kern.type.ndim != 5:
            raise TypeError('kern must be 5D tensor')
        if topgrad.type.ndim != 5:
            raise TypeError('topgrad must be 5D tensor')
        if output.type.ndim != 5:
            raise TypeError('output must be 5D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnConvolutionDescriptor_t':
            raise TypeError('desc must be cudnnConvolutionDescriptor_t')

        alpha = ensure_float(alpha, _one, 'alpha')
        beta = ensure_float(beta, _zero, 'beta')

        return Apply(self, [kern, topgrad, output, desc, alpha, beta],
                     [output.type()])

    def infer_shape(self, node, shape):
        return [shape[2]]




@local_optimizer([GpuDnn3dConv], inplace=True)
def local_dnn3d_conv_inplace(node):
    if type(node.op) != GpuDnn3dConv or node.op.inplace:
        return
    return [GpuDnn3dConv(inplace=True)(*node.inputs)]

@local_optimizer([GpuDnn3dConvGradW], inplace=True)
def local_dnn3d_convgw_inplace(node):
    if type(node.op) != GpuDnn3dConvGradW or node.op.inplace:
        return
    return [GpuDnn3dConvGradW(inplace=True)(*node.inputs)]

@local_optimizer([GpuDnn3dConvGradI], inplace=True)
def local_dnn3d_convgi_inplace(node):
    if type(node.op) != GpuDnn3dConvGradI or node.op.inplace:
        return
    return [GpuDnn3dConvGradI(inplace=True)(*node.inputs)]

optdb.register('local_dnn3d_conv_inplace',
               tensor.opt.in2out(local_dnn3d_conv_inplace,
                                 local_dnn3d_convgw_inplace,
                                 local_dnn3d_convgi_inplace,
                                 name="local_dnn3d_conv_inplace"),
               70.0, 'fast_run', 'inplace', 'gpu', 'cudnn')

@register_opt('cudnn')
@alpha_merge(GpuDnn3dConv, alpha_in=4, beta_in=5, nd=5)
def local_dnn3d_conv_alpha_merge(node, *inputs):
    if not dnn_available() or version() == -1:
        return None
    return [GpuDnn3dConv()(*inputs)]

@register_opt('cudnn')
@alpha_merge(GpuDnn3dConvGradW, alpha_in=4, beta_in=5, nd=5)
def local_dnn3d_convw_alpha_merge(node, *inputs):
    if not dnn_available() or version() == -1:
        return None
    return [GpuDnn3dConvGradW()(*inputs)]

@register_opt('cudnn')
@alpha_merge(GpuDnn3dConvGradI, alpha_in=4, beta_in=5, nd=5)
def local_dnn3d_convi_alpha_merge(node, *inputs):
    if not dnn_available() or version() == -1:
        return None
    return [GpuDnn3dConvGradI()(*inputs)]

@register_opt('cudnn')
@output_merge(GpuDnn3dConv, alpha_in=4, beta_in=5, out_in=2, nd=5)
def local_dnn3d_conv_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [GpuDnn3dConv()(*inputs)]

@register_opt('cudnn')
@output_merge(GpuDnn3dConvGradW, alpha_in=4, beta_in=5, out_in=2, nd=5)
def local_dnn3d_convw_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [GpuDnn3dConvGradW()(*inputs)]

@register_opt('cudnn')
@output_merge(GpuDnn3dConvGradI, alpha_in=4, beta_in=5, out_in=2, nd=5)
def local_dnn3d_convi_output_merge(node, *inputs):
    inputs = inputs[0:2] + (gpu_contiguous(inputs[2]),) + inputs[3:]
    return [GpuDnn3dConvGradI()(*inputs)]

