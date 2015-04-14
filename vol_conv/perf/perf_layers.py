#!/usr/bin/env python
from vol_conv.test_data import generate_test_data
from pylearn2.models.mlp import ConvElemwise, IdentityConvNonlinearity

import theano
import theano.misc.pycuda_init
from vol_conv.convolutions import  compute_out_shape, vectorized_conv
import numpy as np
import argparse
from pylearn2.space import Conv2DSpace
import theano.tensor as T
from perf import perf_func
from vol_conv.layers.blas2d_manuel_conv import ConvElemwiseBlas
from vol_conv.layers.cublas_3d_conv import CuBlasConv3dElemwise
from vol_conv.layers.cudnn_3d_conv import CuDnnConv3dElemwise
#from vol_conv.layers.cudnn_3d_conv import CuDnn
from vol_conv.volumetric_space import Conv3DSpace
from numpy.random import RandomState
# 5 dimensional tensor type for 3d convolutions:
ftensor5 = T.TensorType('float32', (False,)*5)

def perf_layers(inputs_shape, filters_shape):
    # get test data
    rng = RandomState(hash('tobipuma') % 4294967295)
    inputs, filters, bias, inputs_2d, filters_2d = generate_2d_3d_test_data(
        rng, inputs_shape, filters_shape)   
    print_info(inputs_shape, filters_shape)
    function_and_names_2d = create_2d_fprops(inputs_2d.shape, filters_2d, bias)
    function_and_names_3d = create_3d_fprops(inputs.shape, filters, bias)
    reference_result2d = compute_first_result(function_and_names_2d, inputs_2d)
    reference_result3d = compute_first_result(function_and_names_3d, inputs)
    perf_functions(function_and_names_2d, inputs_2d, reference_result2d)
    perf_functions(function_and_names_3d, inputs, reference_result3d)

def generate_2d_3d_test_data(rng, inputs_shape, filters_shape):
    """ Remove time dimension from inputs """
    inputs, filters, bias = generate_test_data(rng,inputs_shape, filters_shape)   
    inputs_2d = inputs[:,:,:,0,:]
    filters_2d = filters[:,:,:,0,:]
    return inputs, filters, bias, inputs_2d, filters_2d

def print_info(inputs_shape, filters_shape):
    output_shape = compute_out_shape(inputs_shape, filters_shape)
    print("Batches/Filters, rows, columns, times, channels")
    print("Input shape  {:s}".format(inputs_shape))
    print("Filter shape {:s}".format(filters_shape))
    print("Output shape {:s}".format(output_shape))
    print("#Inputs  {:7d}".format(np.prod(inputs_shape)))
    print("#Weights {:7d}".format(np.prod(filters_shape)))
    print("#Outputs {:7d}".format(np.prod(output_shape)))
    print("#Multiplications {:7d}".format(
        np.prod(filters_shape) * np.prod(output_shape)))

def create_2d_fprops(inputs_shape, filters, bias):
    functions = []
    filters_flipped = filters[:,::-1,::-1,:]
    conv_2d = create_fprop_layer2d_function(inputs_shape, filters, bias, 
        ConvElemwise)
    conv_2d_blas = create_fprop_layer2d_function(inputs_shape, 
        filters_flipped, bias, ConvElemwiseBlas)
    functions.append({'name': 'conv2d', 'function': conv_2d})
    functions.append({'name': 'conv2dblas', 'function': conv_2d_blas})
    return functions

def create_3d_fprops(inputs_shape, filters, bias):
    functions = []

    conv_3d_blas = create_fprop_layer3d_function(inputs_shape, filters, bias, 
        CuBlasConv3dElemwise)
    conv_3d_cudnn = create_fprop_layer3d_function(inputs_shape, filters, bias, 
        CuDnnConv3dElemwise)
    functions.append({'name': 'conv3dblas', 'function': conv_3d_blas})
    functions.append({'name': 'conv3dcudnn', 'function': conv_3d_cudnn})
    return functions

def create_fprop_layer2d_function(inputs_shape, filters, bias, conv_layer_class):
    # mlp variable needed for setting input space, rng is ignorable (filters 
    # bias are reset later to given values at end of this function)
    mlp = FakeMLP(rng=np.random,batch_size=inputs_shape[0])
    conv_2d_layer = conv_layer_class(output_channels=filters.shape[0], 
        kernel_shape=filters.shape[1:3],
        layer_name='conv_lin', nonlinearity=IdentityConvNonlinearity(),
        irange=0.001, tied_b=True)
    conv_2d_layer.set_mlp(mlp)
    conv_2d_layer.set_input_space(Conv2DSpace(shape=inputs_shape[1:3], 
        num_channels=inputs_shape[3]))
    # convert filters to correct axes (('b', 0, 1, 'c') are test data axes)
    converted_filters = Conv2DSpace.convert_numpy(filters, 
        ('b', 0, 1, 'c'), conv_2d_layer.detector_space.axes)
    conv_2d_layer.set_weights(converted_filters)
    conv_2d_layer.set_biases(bias)
    inputs_2d_theano = T.ftensor4()
    conv2d_result = conv_2d_layer.fprop(inputs_2d_theano)
    conv2d = theano.function([inputs_2d_theano], conv2d_result)
    return conv2d

def create_fprop_layer3d_function(inputs_shape, filters, bias, conv_layer_class):
    # mlp variable needed for setting input space, rng is ignorable (filters 
    # bias are reset to given values at end of this function)
    mlp = FakeMLP(rng=np.random,batch_size=inputs_shape[0])
    conv_3d_input_space = Conv3DSpace(inputs_shape[1:4], 
        num_channels=inputs_shape[4], axes=('b',0,1,2,'c'))
    conv_3d_layer = conv_layer_class(output_channels=filters.shape[0], 
        kernel_shape=filters.shape[1:4], kernel_stride=(1,1,1),
        layer_name='conv3d_lin', nonlinearity=IdentityConvNonlinearity(),
        irange=0.001)
    conv_3d_layer.set_mlp(mlp)
    conv_3d_layer.set_input_space(conv_3d_input_space)
    # convert filters to correct axes (('b', 0, 1, 2, 'c') are test data axes)
    converted_filters = Conv3DSpace.convert_numpy(filters, 
        ('b', 0, 1, 2, 'c'), conv_3d_layer.detector_space.axes)
    conv_3d_layer.set_weights(converted_filters)
    conv_3d_layer.set_biases(bias)
    inputs_3d_theano = ftensor5()
    conv3d_result = conv_3d_layer.fprop(inputs_3d_theano)
    conv3d = theano.function([inputs_3d_theano], conv3d_result)
    return conv3d

def compute_first_result(function_and_names, inputs):
    """ Convenience function for computing a reference result for all tests. """
    function = function_and_names[0]['function']
    result = function(inputs)
    return result

def perf_functions(function_and_names, inputs, reference_result):
    for function_and_name in function_and_names:
        name = function_and_name['name']
        function = function_and_name['function']
        perf_func(name, function, reference_result, inputs)

class FakeMLP():
    """ Fake MLP class with rng and batch size.
    Just because layers always need some parent MLP with 
    rng and batch size"""
    def __init__(self,rng,batch_size):
        self.rng = rng
        self.batch_size = batch_size
    
def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="""Performance experiments for 3d convolution layers.
        Example: ./perf_layers --inputs 15 3 4 5 1 --filters 12 3 4 5 1"""
    )
    parser.add_argument('--inputs', nargs='*', default=[6, 5, 6, 2, 3],
                        help='''Shape of the inputs.''')
    parser.add_argument('--filters', nargs='*', default=[4, 2, 6, 1, 3],
                        help='''Shape of the filters.''')
    args = parser.parse_args()
    # conver to int
    args.inputs = [int(s) for s in args.inputs] 
    args.filters = [int(s) for s in args.filters] 
    return args

if __name__ == '__main__':
    import theano.sandbox.cuda.dnn
    if theano.sandbox.cuda.dnn.dnn_available():
        print(("Cudnn available, using cudnn for theano convolutions "
            "unless theano optimizer flag set to something else"))
    else:
        print("Not using cudnn because:\n{:s}".format(
            theano.sandbox.cuda.dnn.dnn_available.msg))
    args = parse_command_line_arguments()
    perf_layers(args.inputs, args.filters)
    
    #perf/perf_layers.py --inputs 15 80 80 15 3 --filters 12 5 5 5 3