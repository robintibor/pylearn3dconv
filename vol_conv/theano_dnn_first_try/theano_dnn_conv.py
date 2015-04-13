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


def c_set_tensor5d(var, desc, err, fail):
    # TODO when is this even called? remove it maybe? check when other 
    # one is called
    return """
{
    int nbDims = %(var)s->nd; 
    if (nbDims != 5) {
      PyErr_Format(PyExc_RuntimeError,
                   "Number of dimensions should be 5 for 3d convolution, "
                   "instead got %d",
                   nbDims);
      return -1;
    }
    int hostStrides[nbDims];
    // multiply strides in remaining dims in case
    // there is one stride of dimension 0
    int strides_multiplied = 1;
    for (int i = nbDims - 1; i >= 0; i--) {
      hostStrides[i] = CudaNdarray_HOST_STRIDES(var)[i];
      if (hostStrides[i] == 0) {
        hostStrides[i] = strides_multiplied;
      }
      strides_multiplied *= hostStrides[i];
    }
    // TODO: copy necessary?
    int hostDims[nbDims];
    for (int i = 0; i < nbDims; i++) {
      hostDims[i] = CudaNdarray_HOST_DIMS(var)[i];
    }
%(err)s = cudnnSetTensorNdDescriptor(
    %(desc)s, CUDNN_DATA_FLOAT,
    nbDims,
    hostDims,
    hostStrides
);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
         "Could not set tensor5d descriptor: %s"
         "shapes=%d %d %d %d %d, strides=%d %d %d %d %d",
         cudnnGetErrorString(err),
         hostDims[0],
         hostDims[1],
         hostDims[2],
         hostDims[3],
         hostDims[4],
        hostStrides[0],
        hostStrides[1],
        hostStrides[2],
        hostStrides[3],
        hostStrides[4],
    );
  }
}

        """ % dict(var=var, err=err, desc=desc, fail=fail)


class GpuDnnConv3dDesc(GpuOp):
    """This Op builds a 3d convolution descriptor for use in the other
    convolution operations.

    see the doc of :func:`dnn_conv` for a description of the parameters

    """
    __props__ = ('subsample', 'conv_mode')

    def c_headers(self):
        return ['cudnn.h', 'cudnn_helper.h']

    def c_header_dirs(self):
        # TODO check if header dir is doable that way
        return [os.path.dirname(__file__)]
        #return [os.path.dirname(theano.sandbox.cuda.dnn.__file__)]

    def c_libraries(self):
        return ['cudnn']

    def c_compiler(self):
        return NVCC_compiler

    def __init__(self, subsample=(1, 1, 1), conv_mode='conv'):
        # TODO: padding
        assert len(subsample) == 3
        self.subsample = subsample
        assert conv_mode in ('conv', 'cross')
        self.conv_mode = conv_mode

    def make_node(self):

        return Apply(self, [],
                     [CDataType("cudnnConvolutionDescriptor_t")()])

    def c_code(self, node, name, inputs, outputs, sub):
        desc, = outputs
        #TODOndim: make settable?
        num_dims = 5 #for now keep fixed
        
        # TODO pad arr/bordermode
        if self.conv_mode == 'conv':
            conv_flag = 'CUDNN_CONVOLUTION'
        else:
            conv_flag = 'CUDNN_CROSS_CORRELATION'

        return """
{
  cudnnStatus_t err;
  int convDims = 3;
  int padA[3] = {0,0,0};
  int filterStride[3] = {1,1,1}; // TODO: take real stride
  int upscaleA[3]  = { 1,1,1 }; // don't upscale

  if ((err = cudnnCreateConvolutionDescriptor(&%(desc)s)) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate convolution "
                 "descriptor: %%s", cudnnGetErrorString(err));
    %(fail)s
  }
  // TODO: bordermode=padding
  // TODO: just remove if else, check cudnn version somewhere else
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
        filter_stride = self.subsample,)

    def do_constant_folding(self, node):
        # Needed as we do not want to cache this information.
        return False
    
    def c_code_cache_version(self):
        # TODO: set true again
        return None # to prevent cache leak warnings
        #return (2, version())



class GpuDnn3dConv(DnnBase, COp):
    """
    The forward convolution.

    :param image:
    :param kernel:
    :param descr: the convolution descriptor
    """
    __props__ = ('workmem',)

    def __init__(self, workmem=None):
        """
        :param workmem: either 'none', 'small' or 'large'.  Default is
        the value of :attr:`config.dnn.conv.workmem`.
        """
        # TODO: reenable inplace http://deeplearning.net/software/theano/extending/inplace.html
        theano_cuda_dir = os.path.dirname(theano.sandbox.cuda.__file__)
        theano_files = ["dnn_base.c",
            os.path.join(theano_cuda_dir,"dnn_conv_base.c"),
            "dnn_fwd.c"]
        #theano_files = [os.path.join(theano_cuda_dir, f) for f in theano_files]
        COp.__init__(self, theano_files,
                     "APPLY_SPECIFIC(conv_fwd)")
        if workmem is None:
            workmem = config.dnn.conv.workmem
        self.workmem = workmem
        assert self.workmem in ['none', 'small', 'large']

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'workmem'):
            self.workmem = 'none'

    def get_op_params(self):
        if version() == -1:
            alg_def = ('CONV_ALGO', "0")
        else:
            # it seems only this works?
            alg_def = ('CONV_ALGO', 'CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM')
        return [alg_def]

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

    """#TODO: grad
    def grad(self, inp, grads):
        img, kerns, output, desc, alpha, beta = inp
        top, = grads

        top = gpu_contiguous(top)

        d_img = GpuDnnConvGradI()(kerns, top, gpu_alloc_empty(*img.shape), desc)
        d_kerns = GpuDnnConvGradW()(img, top, gpu_alloc_empty(*kerns.shape), desc)
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)

        return [d_img * alpha, d_kerns * alpha, top * beta,
                DisconnectedType()(), d_alpha, d_beta]

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [1], [0], [1], [1]]
    """
    
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

    # TODO: reenable constant folding
    def do_constant_folding(self, node):
        # Needed as we do not want to cache this information.
        return False
    
    # TODO: reenable caching
    def c_code_cache_version(self):
        # TODO: set true again
        from time import time
        return int(time() * 100)# always recompile
        #return (2, version())

