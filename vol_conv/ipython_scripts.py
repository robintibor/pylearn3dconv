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
from pylearn2.models.mlp import MLP, Softmax
from pylearn2.format.target_format import OneHotFormatter
from numpy.random import RandomState


import theano
import theano.sandbox.cuda.dnn as cdnn
from theano.sandbox.cuda.basic_ops import (as_cuda_ndarray_variable,
                                           host_from_gpu,
                                           gpu_contiguous, HostFromGpu,
                                           gpu_alloc_empty)

import numpy as np
import theano_dnn_first_try.theano_dnn_conv as owndnn

ftensor5 = T.TensorType('float32', (False,)*5)

rng = RandomState(hash('manuelloewe') % 4294967296)
inputs = ftensor5()
filters = ftensor5()
inputs = gpu_contiguous(inputs)
filters = gpu_contiguous(filters)
desc = owndnn.GpuDnnConv3dDesc(subsample=(1,1,1), conv_mode='cross')()

desc_op = desc.owner.op
out_shp = owndnn.GpuDnn3dConv.get_out_shape(inputs.shape, filters.shape,
                                   desc_op.subsample)

out = gpu_alloc_empty(*out_shp)
#680710144#680710144#680710144
# no reference: 762105856#762105856
conv_result = owndnn.GpuDnn3dConv()(inputs, filters, out, desc)

conv_result_func = theano.function([inputs, filters], conv_result, mode='DebugMode')
real_inputs = rng.normal(size=(5,3,4,3,1)).astype(np.float32)
real_filters = rng.normal(size=(2,3,3,2,1)).astype(np.float32)

real_inputs = rng.normal(size=(3,3,3,3,3)).astype(np.float32)
real_filters = rng.normal(size=(3,3,3,3,3)).astype(np.float32)
#real_inputs = np.ones((1,2,2,2,2)).astype(np.float32)
#real_filters = np.ones((1,2,2,2,2)).astype(np.float32)
#real_inputs[0,0,0,1,0] = 2
bias = np.zeros(real_filters.shape[0])
correct_result = vectorized_conv(real_inputs, real_filters, bias)
result=conv_result_func(real_inputs, real_filters)
print np.array(result)
print correct_result