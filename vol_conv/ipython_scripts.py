import theano.tensor as T
import theano
import theano.sandbox.cuda.dnn as cdnn
import numpy as np
import theano_dnn_first_try.theano_dnn_conv as owndnn
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)

from convolutions import vectorized_conv
def test_one():
    img = T.ftensor4()
    kernel = T.ftensor4()
    conv_result = cdnn.dnn_conv(img, kernel)
    conv_func = theano.function([img, kernel], conv_result)
    result= conv_func(np.random.normal(size=(3,4,5,6)).astype('float32'), 
        np.random.normal(size=(2,4,3,5)).astype('float32'))
    print np.array(result).shape

def test_two():
    img_shape = T.lvector()
    kernel_shape = T.lvector()
    a = owndnn.GpuDnnConv3dDesc('valid')
    desc_result = a(img_shape, kernel_shape)
    conv_desc_func = theano.function([img_shape, kernel_shape], desc_result)
    desc = conv_desc_func([3,4],[2,2])

from vol_conv.volumetric_space import Conv3DSpace
from vol_conv.layers.theano_3d_conv import Theano3dConv3dElemwise
import numpy as np
from pylearn2.models.mlp import IdentityConvNonlinearity
import theano
import theano.tensor as T
from vol_conv.volumetric_dense_design_matrix import VolumetricDenseDesignMatrix
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.models.mlp import MLP, Softmax, ConvElemwise
from pylearn2.format.target_format import OneHotFormatter
from numpy.random import RandomState
from pylearn2.space import Conv2DSpace
from vol_conv.layers.blas2d_manuel_conv import ConvElemwiseBlas
from vol_conv.layers.cublas_3d_conv import CuBlasConv3dElemwise
from vol_conv.layers.cudnn_3d_conv import CuDnnConv3dElemwise
from vol_conv.perf.perf_layers import create_fprop_layer_3d_symbolic
from vol_conv.test_data import generate_test_data
import theano.sandbox.cuda
import theano
import theano.sandbox.cuda.dnn as cdnn
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)
from theano.sandbox.cuda.dnn import GpuDnnConv, GpuDnnConvDesc
from numpy.random import RandomState
from vol_conv.theano_dnn_first_try.theano_dnn_conv import GpuDnn3dConv, GpuDnnConv3dDesc

import numpy as np
import theano_dnn_first_try.theano_dnn_conv as owndnn
ftensor5 = T.TensorType('float32', (False,)*5)
class FakeMLP():
    def __init__(self,rng,batch_size):
        self.rng = rng
        self.batch_size = batch_size
        
        
# get small input
rng = RandomState(np.uint32(hash('tobiderpuma')))
inputs_shape = [5,8,4,7,3]
inputs_shape = [5,2,4,7,3]
filters_shape = [6,2,3,5,3]


inputs, filters, bias = generate_test_data(rng, inputs_shape, filters_shape)
#bias *= 0
# compute gradient for Cublas
# do it twice, compare result
x = T.dscalar('x')
inputs_theano = ftensor5()

conv_result = create_fprop_layer_3d_symbolic(inputs_shape, filters, bias, CuBlasConv3dElemwise, inputs_theano)
cost = T.sum(conv_result)
conv_gradient = T.grad(cost, inputs_theano)
grad_func = theano.function([inputs_theano], conv_gradient)

correct_result = grad_func(inputs)

inputs_theano_cudnn = ftensor5()
inputs_theano_cudnn_contiguous = gpu_contiguous(inputs_theano_cudnn)
conv_result_cudnn = create_fprop_layer_3d_symbolic(inputs_shape, filters, bias, CuDnnConv3dElemwise, 
                                                   inputs_theano_cudnn_contiguous)
cost_cudnn = T.sum(conv_result_cudnn)
cost_cudnn = gpu_contiguous(cost_cudnn)
conv_dnn_gradient = T.grad(cost_cudnn, inputs_theano_cudnn)
grad_func_cudnn = theano.function([inputs_theano_cudnn], conv_dnn_gradient)

cudnn_result = grad_func_cudnn(inputs)
assert np.sum(np.square(cudnn_result- correct_result)) < 1e-4
print "assertion passed results same"