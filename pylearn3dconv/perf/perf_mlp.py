from pylearn3dconv.test import ftensor5
from numpy.random import RandomState
import numpy as np
from pylearn3dconv.volumetric_space import Conv3DSpace
from pylearn3dconv.layers.variants import CuDnnConv3dElemwise
from pylearn2.models.mlp import Softmax, MLP, IdentityConvNonlinearity
import theano.tensor as T
from pylearn3dconv.perf import perf_func_print_results
import theano
from theano.sandbox.cuda.basic_ops import gpu_contiguous

def perf_mlp():
    rng = RandomState(np.uint32(hash('perfthemlp')))
    # b 0 1 2 c format
    inputs_shape = [32, 80, 80, 20, 3]
    filters_shape = [32, 5, 5, 5, 3]
    # generate mlp grad
    # generate inputs
    # perf....
    inputs = rng.normal(size=inputs_shape).astype(np.float32)
    mlp_grad_func = create_mlp_grad_func(inputs_shape, filters_shape)
    mlp_grad_func(inputs)
    perf_func_print_results("Gradient Single layer", mlp_grad_func, None, inputs)
    

def create_mlp_grad_func(inputs_shape, filters_shape):
    inputs = ftensor5()
    mlp = construct_model(inputs_shape, filters_shape)
    result = mlp.fprop(inputs)
    cost = T.sum(result)
    grad = T.grad(cost, inputs)
    grad = gpu_contiguous(grad)
    grad_func = theano.function([inputs], grad)
    return grad_func
    

def construct_model(inputs_shape, filters_shape):
    conv_3d_input_space = Conv3DSpace(inputs_shape[1:4], 
        num_channels=inputs_shape[4], axes=('b',0,1,2,'c'))
    conv_3d_layer = CuDnnConv3dElemwise(output_channels=filters_shape[0], 
        kernel_shape=filters_shape[1:4], kernel_stride = (1,1,1),
        layer_name='conv3d_lin', nonlinearity=IdentityConvNonlinearity(),
        irange=0.001, pool_type=None, pool_shape=None,
        pool_stride=None)
    softmax_layer = Softmax(max_col_norm=2, layer_name='y',
        n_classes=2, istdev=.05)
    mlp = MLP(input_space=conv_3d_input_space, layers=[conv_3d_layer])
    return mlp


if __name__ == "__main__":
    perf_mlp()