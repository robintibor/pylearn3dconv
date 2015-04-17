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
  int padA[3] = {0,0,0}; // TODELAY: no padding for now, could reenable that
  int filterStride[3] = {%(filter_stride)s}; // TODO: take real stride
  int upscaleA[3]  = { 1,1,1 }; // don't upscale

  if ((err = cudnnCreateConvolutionDescriptor(&%(desc)s)) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate convolution "
                 "descriptor: %%s", cudnnGetErrorString(err));
    %(fail)s
  }
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
        # transform subsample tuple to x,y,z string like 1,1,1
        filter_stride = str(tuple(self.subsample)).strip('()'))

    def do_constant_folding(self, node):
        
        # Todo check if this is from original, if not remove it again
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
    __props__ = ()

    def __init__(self,  inplace=False):
        # TODO: reenable inplace optimizations http://deeplearning.net/software/theano/extending/inplace.html
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




class GpuDnnPool3dDesc(GpuOp):
    """
    This Op builds a 3d pooling descriptor for use in the other
    pooling operations.

    :param ws: windows size (x,y,z)
    :param stride: (x,y,z)
    :param mode: 'max' or 'average'
    :param pad: (padX, padY, padz) padding information.
        padX is the size of the left and right borders,
        padY is the size of the top and bottom borders.
    """
    __props__ = ('ws', 'stride', 'mode', 'pad')

    def c_headers(self):
        return ['cudnn.h', 'cudnn_helper.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__)]

    def c_libraries(self):
        return ['cudnn']

    def c_compiler(self):
        return NVCC_compiler

    def do_constant_folding(self, node):
        return False

    def __init__(self, ws=(1, 1, 1), stride=(1, 1, 1), mode='max', pad=(0, 0, 0)):
        if version() == -1:
            raise RuntimeError("CuDNN Nd-pooling requires CuDNN v2")
        assert mode in ('max', 'average')
        self.mode = mode
        assert len(ws) == 3
        self.ws = ws
        assert len(stride) == 3
        self.stride = stride
        assert len(pad) == 3
        self.pad = pad

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, 'pad'):
            self.pad = (0, 0)

    def make_node(self):
        if version() == -1:
            raise RuntimeError("CuDNN Nd-pooling requires CuDNN v2")

        return Apply(self, [],
                     [CDataType("cudnnPoolingDescriptor_t")()])

    def c_code(self, node, name, inputs, outputs, sub):
        desc, = outputs

        if self.mode == 'max':
            mode_flag = 'CUDNN_POOLING_MAX'
        elif self.mode == "average":
            mode_flag = 'CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING'
        else:
            raise NotImplementedError("Unsupported pooling model.")

        return """
{
  cudnnStatus_t err;
  int nbDims = 3;
  if ((err = cudnnCreatePoolingDescriptor(&%(desc)s)) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate pooling "
                 "descriptor: %%s", cudnnGetErrorString(err));
    %(fail)s
  }
  int windowDimA[] = {%(window_size)s};
  int paddingA[] = {%(padding)s};
  int strideA[] = {%(strides)s};
  err = cudnnSetPoolingNdDescriptor(
  %(desc)s,
  %(mode_flag)s,
  nbDims,
  windowDimA,
  paddingA,
  strideA
  );
  
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: %%s",
                 cudnnGetErrorString(err));
    %(fail)s
  }
}
""" % dict(name=name, desc=desc, mode_flag=mode_flag, fail=sub['fail'],
           # transform tuples to x,y,z strings
           window_size=str(tuple(self.ws)).strip('()'),
           padding=str(tuple(self.pad)).strip('()'),
           strides=str(tuple(self.stride)).strip('()'))

    
    # TODO: reenable caching
    def c_code_cache_version(self):
        # TODO: set true again
        from time import time
        return int(time() * 100)# always recompile
        #return (2, version())

""
def c_set_tensor5d(var, desc, err, fail):
    return """
{
    int nbDims = %(var)s->nd;
    if (nbDims != 5) {
      PyErr_Format(PyExc_RuntimeError,
                   "Number of dimensions should be 5 for 3d convolution, "
                   "instead got %%d",
                   nbDims);
      return -1;
    }
    int hostStrides[nbDims];
    // multiply dims of remaining dims in case
    // there is one stride of dimension 0
    int dims_multiplied = 1;
    for (int i = nbDims - 1; i >= 0; i--) {
      hostStrides[i] = CudaNdarray_HOST_STRIDES(%(var)s)[i];
      if (hostStrides[i] == 0) {
        hostStrides[i] = dims_multiplied;
      }
      dims_multiplied *= CudaNdarray_HOST_DIMS(%(var)s)[i];
    }
    // TODO: copy necessary?
    int hostDims[nbDims];
    for (int i = 0; i < nbDims; i++) {
      hostDims[i] = CudaNdarray_HOST_DIMS(%(var)s)[i];
    }

    cudnnStatus_t err = cudnnSetTensorNdDescriptor(
    %(desc)s, CUDNN_DATA_FLOAT,
    nbDims,
    hostDims,
    hostStrides
  );

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
         "Could not set tensor5d descriptor: %%s"
         "shapes=%%d %%d %%d %%d %%d, strides=%%d %%d %%d %%d %%d",
         cudnnGetErrorString(err),
         CudaNdarray_HOST_DIMS(%(var)s)[0],
         CudaNdarray_HOST_DIMS(%(var)s)[1],
         CudaNdarray_HOST_DIMS(%(var)s)[2],
         CudaNdarray_HOST_DIMS(%(var)s)[3],
         CudaNdarray_HOST_DIMS(%(var)s)[4],
         hostStrides[0],
         hostStrides[1],
         hostStrides[2],
         hostStrides[3],
         hostStrides[4]
    );
    %(fail)s
  }
}

        """ % dict(var=var, err=err, desc=desc, fail=fail)




class GpuDnn3dPool(DnnBase):
    """
    Pooling.

    :param img: the image 5d tensor.
    :param desc: the pooling descriptor.
    """
    __props__ = ()

    def make_node(self, img, desc):
        img = as_cuda_ndarray_variable(img)
        if img.type.ndim != 5:
            raise TypeError('img must be 5D tensor')

        if not isinstance(desc.type, CDataType) \
                or desc.type.ctype != 'cudnnPoolingDescriptor_t':
            raise TypeError('desc must be cudnnPoolingDescriptor_t')

        return Apply(self, [img, desc],
                     [img.type()])

    def infer_shape(self, node, shape):
        desc = node.inputs[1].owner.op
        kh, kw, kd = desc.ws
        sh, sw, sd = desc.stride
        padh, padw, padd = desc.pad
        return [(
            shape[0][0],
            shape[0][1],
            (shape[0][2] + 2*padh - kh)//sh + 1,
            (shape[0][3] + 2*padw - kw)//sw + 1,
            (shape[0][4] + 2*padd - kd)//sd + 1
        )]

    def c_support_code_struct(self, node, name):
        return """
cudnnTensorDescriptor_t input%(name)s;
cudnnTensorDescriptor_t output%(name)s;
""" % dict(name=name)

    def c_init_code_struct(self, node, name, sub):
        return """
cudnnStatus_t err%(name)s;
input%(name)s = NULL;
output%(name)s = NULL;
if ((err%(name)s = cudnnCreateTensorDescriptor(&input%(name)s)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor5d descriptor "
               "(inp): %%s", cudnnGetErrorString(err%(name)s));
  %(fail)s
}
if ((err%(name)s = cudnnCreateTensorDescriptor(&output%(name)s)) != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_MemoryError, "could not allocate tensor5d descriptor "
               "(out): %%s", cudnnGetErrorString(err%(name)s));
  %(fail)s
}
""" % dict(name=name, fail=sub['fail'])

    def c_cleanup_code_struct(self, node, name):
        return """
if (input%(name)s != NULL) { cudnnDestroyTensorDescriptor(input%(name)s); }
if (output%(name)s != NULL) { cudnnDestroyTensorDescriptor(output%(name)s); }
""" % dict(name=name)

    def c_code(self, node, name, inputs, outputs, sub):
        desc = inputs[1]
        out, = outputs

        set_in = c_set_tensor5d(inputs[0], "input" + str(name),
                                'err' + name, sub['fail'])

        set_out = c_set_tensor5d(out, "output" + str(name),
                                 'err' + name, sub['fail'])

        return """
cudnnStatus_t err%(name)s;

int %(out)s_dims[5];

cudnnPoolingMode_t mode;
int nbDimsRequested = 3;
int windowDimA[nbDimsRequested];
int paddingA[nbDimsRequested];
int strideA[nbDimsRequested];
int nbDims = -1;

if (!CudaNdarray_is_c_contiguous(%(input)s)) {
  PyErr_SetString(PyExc_ValueError, "Only contiguous inputs are supported.");
  %(fail)s
}

%(set_in)s


err%(name)s = cudnnGetPoolingNdDescriptor(%(desc)s, nbDimsRequested, &mode,
    &nbDims, windowDimA, paddingA, strideA);

if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError,
               "GpuDnn3dPool: error doing cudnnGetPoolingNdDescriptor operation: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}

%(out)s_dims[0] = CudaNdarray_HOST_DIMS(%(input)s)[0];
%(out)s_dims[1] = CudaNdarray_HOST_DIMS(%(input)s)[1];
%(out)s_dims[2] = (CudaNdarray_HOST_DIMS(%(input)s)[2] + 
    (paddingA[0]*2) - windowDimA[0]) / strideA[0] + 1;
%(out)s_dims[3] = (CudaNdarray_HOST_DIMS(%(input)s)[3] + 
    (paddingA[1]*2) - windowDimA[1]) / strideA[1] + 1;
%(out)s_dims[4] = (CudaNdarray_HOST_DIMS(%(input)s)[4] + 
    (paddingA[2]*2) - windowDimA[2]) / strideA[2] + 1;

if (CudaNdarray_prep_output(&%(out)s, 5, %(out)s_dims) != 0)
{
  %(fail)s
}

%(set_out)s

// PRINTING START
{
    cudnnDataType_t dataType;
    nbDims = -1;
    nbDimsRequested = 5;
    int dimA[nbDimsRequested];
    int strideA[nbDimsRequested];
 cudnnGetTensorNdDescriptor(%(input_desc)s,
        5,
        &dataType,
        &nbDims,
        dimA,
        strideA);
    printf("Input descriptor\\n");
    printf("Float: %%d\\n", dataType == CUDNN_DATA_FLOAT);
    printf("nbDims: %%d\\n", nbDims);
    printf("Dimensions: ");
    for (int i = 0; i < nbDims; ++i) {
      printf("%%d ", dimA[i]);
    }
    printf("\\n");
    printf("Strides: ");
    for (int i = 0; i < nbDims; ++i) {
      printf("%%d ", strideA[i]);
    }
    printf("\\n");
    printf("\\n");
    
    
 cudnnGetTensorNdDescriptor(%(output_desc)s,
        5,
        &dataType,
        &nbDims,
        dimA,
        strideA);
    printf("Output descriptor\\n");
    printf("Float: %%d\\n", dataType == CUDNN_DATA_FLOAT);
    printf("nbDims: %%d\\n", nbDims);
    printf("Dimensions: ");
    for (int i = 0; i < nbDims; ++i) {
      printf("%%d ", dimA[i]);
    }
    printf("\\n");
    printf("Strides: ");
    for (int i = 0; i < nbDims; ++i) {
      printf("%%d ", strideA[i]);
    }
    printf("\\n");
    printf("\\n");
    
    
    cudnnPoolingMode_t mode;
    nbDimsRequested =3;
    int padA[nbDimsRequested];
    cudnnGetPoolingNdDescriptor(%(desc)s,
    nbDimsRequested,
    &mode,
    &nbDims,
    dimA,
    padA,
    strideA);
    printf("Pooling descriptor\\n");
    printf("nbDims: %%d\\n", nbDims);
    printf("Shape: ");
    for (int i = 0; i < nbDims; ++i) {
      printf("%%d ", dimA[i]);
    }
    printf("\\n");
    printf("Padding: ");
    for (int i = 0; i < nbDims; ++i) {
      printf("%%d ", padA[i]);
    }
    printf("\\n");
    printf("Strides: ");
    for (int i = 0; i < nbDims; ++i) {
      printf("%%d ", strideA[i]);
    }
    printf("\\n");
    
    
    }
// PRINTING END

{
const float alpha = 1;
const float beta = 0;
err%(name)s = cudnnPoolingForward(
_handle,
%(desc)s,
&alpha,
%(input_desc)s, CudaNdarray_DEV_DATA(%(input)s),
&beta,
%(output_desc)s, CudaNdarray_DEV_DATA(%(out)s)
);
}


if (err%(name)s != CUDNN_STATUS_SUCCESS) {
  PyErr_Format(PyExc_RuntimeError,
               "GpuDnnPool: error doing cudnnPoolingForward operation: %%s",
               cudnnGetErrorString(err%(name)s));
  %(fail)s
}
""" % dict(out=out, desc=desc, fail=sub['fail'],
           name=name, set_in=set_in,
           set_out=set_out, input=inputs[0],
           input_desc="input"+name,
           output_desc="output"+name)
    """ TODO: gradient! :)
    def grad(self, inp, grads):
        img, desc = inp
        grad, = grads

        grad = gpu_contiguous(grad)

        out = self(img, desc)

        g_out = GpuDnnPoolGrad()(img, out, grad, desc)

        return g_out, theano.gradient.DisconnectedType()()

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [0]]"""

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






