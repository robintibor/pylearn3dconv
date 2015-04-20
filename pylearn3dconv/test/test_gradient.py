from pylearn3dconv.volumetric_space import Conv3DSpace

from pylearn3dconv.layers.conv_transformers import (CuDnn3dConv, CuBlas3dConv,
    Theano3dConv)
from pylearn3dconv.layers.pool_transformers import CudnnPoolTransformer
import numpy as np
from pylearn2.models.mlp import IdentityConvNonlinearity
import theano
import theano.tensor as T
from pylearn2.models.mlp import MLP
from numpy.random import RandomState
from pylearn3dconv.test import generate_test_data, ftensor5
from pylearn3dconv.test.test_dnn_3dconvolution import compute_reference_result
from pylearn3dconv.layers.base import Conv3dElemwise

def test_gradients():
    #b 0 1 2 c format
    inputs_shape = [100,14,10,8,3]
    filters_shape = [11,4,3,2,3]

    print("No stride no pooling (1st layer)")
    kernel_stride = [1, 1, 1]
    pool_type = None
    pool_shape = (1,1,1)
    pool_stride = (1,1,1)
    
    compare_gradient_results(inputs_shape, filters_shape, kernel_stride,
        pool_type, pool_shape, pool_stride)
    # Then with stride
    print("Stride and no pooling (1st layer)")
    kernel_stride = [2, 1, 2]
    pool_type = None
    pool_shape = (1,1,1)
    pool_stride = (1,1,1)
    compare_gradient_results(inputs_shape, filters_shape, kernel_stride,
        pool_type, pool_shape, pool_stride)
    
    # With stride and fake pooling expect same results as without pooling
    print("Stride and fake pooling (1st layer)")
    kernel_stride = (2, 1, 2)
    pool_type = 'max'
    pool_shape = (1,1,1)
    pool_stride = (1,1,1)
    compare_gradient_results(inputs_shape, filters_shape, kernel_stride,
        pool_type, pool_shape, pool_stride)

    print("Real pooling (1st layer)")
    pool_type = 'max'
    pool_shape = (2,2,2)
    pool_stride = (1,1,1)
    kernel_stride = [2, 1, 2]
    compare_gradient_results(inputs_shape, filters_shape, kernel_stride,
        pool_type, pool_shape, pool_stride)

def compare_gradient_results(inputs_shape, filters_shape, kernel_stride, 
        pool_type, pool_shape, pool_stride):
    # a great seed is half the work :)
    rng = RandomState(hash('tobipuma') % 4294967295)
    inputs, filters_1, filters_2, bias = generate_test_data_two_layers(rng, 
        inputs_shape, filters_shape,  kernel_stride, 
        pool_type, pool_shape, pool_stride)
    reference_result = compute_reference_result(inputs, filters_1, 
        filters_2, bias, kernel_stride, pool_type, pool_shape, pool_stride)
    #Theano3dConv3dElemwise, CuBlasConv3dElemwise, 
    conv_classes = [CuDnn3dConv, Theano3dConv] # cublas is reference
    for conv_class in conv_classes:
        compare_results_for_layer(inputs, filters_1, filters_2, bias,
            kernel_stride, pool_type, pool_shape, pool_stride, conv_class,
            reference_result)

def generate_test_data_two_layers(rng, inputs_shape, filters_shape,
    kernel_stride, pool_type, pool_shape, pool_stride):
    inputs, filters_1, bias = generate_test_data(rng, 
        inputs_shape, filters_shape)
    in_2 = filters_shape[0:4] + [filters_shape[0]]
    filters_2 = rng.normal(size=in_2).astype(np.float32)
    return inputs, filters_1, filters_2, bias

def compute_reference_result(inputs, filters_1, filters_2, bias,
        kernel_stride, pool_type, pool_shape, pool_stride):
    mlp = construct_model(inputs.shape, filters_1, filters_2, bias,
        kernel_stride, pool_type, pool_shape, pool_stride,  CuBlas3dConv)
    grad_func = compute_grad_func(mlp)
    result = grad_func(inputs)
    return result

def compare_results_for_layer(inputs, filters_1, filters_2, bias, kernel_stride,
            pool_type, pool_shape, pool_stride, layer_class, reference_result):
    mlp = construct_model(inputs.shape, filters_1, filters_2, bias, 
        kernel_stride, pool_type, pool_shape, pool_stride,  layer_class)
    grad_func = compute_grad_func(mlp)
    result = grad_func(inputs)
    assert np.allclose(result, reference_result, rtol=0, atol=1e-3)
    print layer_class.__name__ + " - Ok."
    
def construct_model(inputs_shape, filters_1, filters_2, bias, kernel_stride,
    pool_type, pool_shape, pool_stride,  conv_class):
    """ Two layer model """
    kernel_shape = filters_1.shape[1:4]
    conv_3d_input_space = Conv3DSpace(inputs_shape[1:4], 
        num_channels=inputs_shape[4], axes=('b',0,1,2,'c'))
    conv_3d_layer = Conv3dElemwise(output_channels=filters_1.shape[0], 
        kernel_shape=kernel_shape, kernel_stride = kernel_stride,
        layer_name='conv3d_lin', nonlinearity=IdentityConvNonlinearity(),
        conv_transformer_class=conv_class,
        pool_transformer_class=CudnnPoolTransformer,
        irange=0.001, pool_type=pool_type, pool_shape=pool_shape,
        pool_stride=pool_stride)
    conv_3d_layer_2 = Conv3dElemwise(output_channels=filters_1.shape[0], 
        kernel_shape=kernel_shape, kernel_stride = (1,1,1),
        layer_name='conv3d_lin2', nonlinearity=IdentityConvNonlinearity(),
        conv_transformer_class=conv_class,
        pool_transformer_class=CudnnPoolTransformer,
        irange=0.001, pool_type=None)
    mlp = MLP(input_space=conv_3d_input_space, layers=[conv_3d_layer,
        conv_3d_layer_2])
    # convert filters to correct axes (('b', 0, 1, 2, ' c') are test data axes)
    #filters2 = filters[:, ]
    converted_filters_1 = Conv3DSpace.convert_numpy(filters_1, 
        ('b', 0, 1, 2, 'c'), conv_3d_layer.detector_space.axes)
    conv_3d_layer.set_weights(converted_filters_1)
    conv_3d_layer.set_biases(bias)
    converted_filters2 = Conv3DSpace.convert_numpy(filters_2, 
        ('b', 0, 1, 2, 'c'), conv_3d_layer.detector_space.axes)
    conv_3d_layer_2.set_weights(converted_filters2)
    conv_3d_layer_2.set_biases(bias)
    return mlp 

def compute_grad_func(mlp):
    inputs_theano = ftensor5()
    output = mlp.fprop(inputs_theano)
    cost = T.sum(output)
    grad = T.grad(cost, inputs_theano)
    grad_func = theano.function([inputs_theano], grad)
    return grad_func

if __name__ == '__main__':
    test_gradients()