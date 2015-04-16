import theano.tensor as T
import theano
import theano.sandbox.cuda.dnn as cdnn
import theano.misc.pycuda_init
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

def old_test_gradient():
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
        

def max_pool_numpy(inputs, pool_shape, pool_stride):
    assert (len(inputs.shape) ==4) # for now restrict to 2d images 
    # (in bc01 format)
    output_shape = [(inputs.shape[i + 2] - pool_shape[i]) // pool_stride[i] + 1 
        for i in xrange(len(pool_shape))]
    output_shape = list(inputs.shape[0:2]) + output_shape
    output = np.ones(output_shape) * np.nan
    for batch_i in xrange(output_shape[0]):
        for chan_i in xrange(output_shape[1]):
            for x in xrange(output_shape[2]):
                input_x = x * pool_stride[0]
                for y in xrange(output_shape[3]):
                    input_y = y * pool_stride[1]
                    max_value = np.max(inputs[batch_i, chan_i,
                        input_x:input_x+pool_shape[0], 
                        input_y:input_y+pool_shape[1]])
                    output[batch_i,chan_i,x,y] = max_value
    return output

def max_pool_3d_numpy(inputs, pool_shape, pool_stride):
    assert (len(inputs.shape) ==5) # for now restrict to 3d images
    assert(len(pool_shape) ==3)
    assert(len(pool_stride) ==3)

    # in b c 0 1 2 format
    output_shape = [(inputs.shape[i + 2] - pool_shape[i]) // pool_stride[i] + 1 
        for i in xrange(len(pool_shape))]
    output_shape = list(inputs.shape[0:2]) + output_shape
    output = np.ones(output_shape) * np.nan
    for batch_i in xrange(output_shape[0]):
        for chan_i in xrange(output_shape[1]):
            for x in xrange(output_shape[2]):
                input_x = x * pool_stride[0]
                for y in xrange(output_shape[3]):
                    input_y = y * pool_stride[1]
                    for z in xrange(output_shape[4]):
                        input_z = z * pool_stride[2]
                        max_value = np.max(inputs[batch_i, chan_i,
                            input_x:input_x+pool_shape[0], 
                            input_y:input_y+pool_shape[1],
                            input_z:input_z+pool_shape[2]])
                        output[batch_i,chan_i,x,y,z] = max_value
    return output

def max_pool_3d_2d(inputs, pool_shape, pool_stride):
    dnn_pool2d = create_dnn_pool_func(pool_shape[0:2], pool_stride[0:2])
    output_shape = [(inputs.shape[i + 2] - pool_shape[i]) // pool_stride[i] + 1 
        for i in xrange(len(pool_shape))]
    output_shape = list(inputs.shape[0:2]) + output_shape
    first_2d_pooled_output = np.ones(output_shape[0:4] + [inputs.shape[4]]).astype('float32')
    for z in range(inputs.shape[4]):
        pooled_slice = dnn_pool2d(inputs[:,:,:,:,z])
        first_2d_pooled_output[:,:,:,:,z] = pooled_slice
    # now 1d-pool over last dimension...
    # coudl use first or second dimension as input fo pool1d..
    output = np.ones(output_shape).astype(np.float32) * np.nan
    dnn_pool1d = create_dnn_pool_func(pool_shape=(1, pool_shape[2]), 
        pool_stride=(1, pool_stride[2]))
    for y in range(first_2d_pooled_output.shape[3]):
        pooled_slice = dnn_pool1d(first_2d_pooled_output[:,:,:,y,:])
        output[:,:,:,y,:] = pooled_slice
    return output
    
def create_dnn_pool_func(pool_shape, pool_stride):
    input_dnn = T.ftensor4()
    outputdnn =  dnn_pool(input_dnn, ws=pool_shape, 
        stride=pool_stride, mode='max')
    max_pool_dnn_func = theano.function([input_dnn], outputdnn)
    return max_pool_dnn_func

from pylearn2.models.mlp import max_pool  
from theano.sandbox.cuda.dnn import dnn_pool  
rng = RandomState(np.uint32(hash('tobiderpuma')))
"""    
input_shape = [5,2,8,7] # bc01
pool_shape = (3,4)
pool_stride = (3,1)
# get small input
inputs = rng.normal(size=input_shape).astype(np.float32)
input_pylearn = T.ftensor4()
output =  max_pool(input_pylearn, pool_shape=pool_shape, 
    pool_stride=pool_stride, image_shape=inputs.shape[2:4], try_dnn=False)

max_pool_pylearn_func = theano.function([input_pylearn], output)
input_dnn = T.ftensor4()
outputdnn =  dnn_pool(input_dnn, ws=pool_shape, 
    stride=pool_stride, mode='max')
max_pool_dnn_func = theano.function([input_dnn], outputdnn)

reference_result = max_pool_pylearn_func(inputs)
dnn_result = max_pool_dnn_func(inputs)
numpy_result = max_pool_numpy(inputs, pool_shape, pool_stride)

assert np.sum(np.square(dnn_result - numpy_result)) < 1e-4
 """

input_shape = [5,2,8,7,6]#bc012
pool_shape = (3,4,2)
pool_stride = (3,1,4)
inputs = rng.normal(size=input_shape).astype(np.float32)
numpy_result = max_pool_3d_numpy(inputs, pool_shape, pool_stride)
print "numpy shape"
print numpy_result.shape
dnn_3d_2d_result = max_pool_3d_2d(inputs, pool_shape, pool_stride)
print "dnn"
print dnn_3d_2d_result.shape
assert np.sum(np.square(dnn_3d_2d_result - numpy_result)) < 1e-4