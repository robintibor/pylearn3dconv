#!/usr/bin/env python
from pylearn2.models.mlp import ConvElemwise, IdentityConvNonlinearity

import theano
import theano.misc.pycuda_init
from test_data import generate_test_data
from convolutions import  compute_out_shape
import numpy as np
import argparse
from pylearn2.space import Conv2DSpace
import theano.tensor as T
from perf import perf_func
from layers.blas2d_manuel_conv import ConvElemwiseBlas

class FakeMLP():
    def __init__(self,rng,batch_size):
        self.rng = rng
        self.batch_size = batch_size

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

def create_fprop_functions(inputs_shape, filters_shape):
    functions = []
    conv_2d = create_fprop_layer_function(inputs_shape, filters_shape, 
        ConvElemwise)
    conv_2d_blas = create_fprop_layer_function(inputs_shape, filters_shape, 
        ConvElemwiseBlas)
    functions.append({'name': 'conv2d', 'function': conv_2d})
    functions.append({'name': 'conv2dblas', 'function': conv_2d_blas})
    return functions

def create_fprop_layer_function(inputs_shape, filters_shape, conv_layer_class):
    # mlp variable needed for setting input space
    mlp = FakeMLP(rng=np.random,batch_size=inputs_shape[0])
    conv_2d_layer = conv_layer_class(output_channels=filters_shape[0], 
        kernel_shape=filters_shape[1:3],
        layer_name='conv_lin', nonlinearity=IdentityConvNonlinearity(),
        irange=0.001)
    conv_2d_layer.set_mlp(mlp)
    conv_2d_layer.set_input_space(Conv2DSpace(shape=inputs_shape[1:3], 
        num_channels=inputs_shape[4]))
    inputs_2d_theano = T.ftensor4()
    conv2d_result = conv_2d_layer.fprop(inputs_2d_theano)
    conv2d = theano.function([inputs_2d_theano], conv2d_result)
    return conv2d
    
    
def perf_functions(function_and_names, inputs, filters, bias):
    inputs_2d = inputs[:,:,:,0,:] # remove time dimension(axis 3)
    for function_and_name in function_and_names:
        name = function_and_name['name']
        function = function_and_name['function']
        if name.startswith('conv2d'):
            perf_func(name, function, None, inputs_2d)
            
    
def perf_layers(inputs_shape, filters_shape):
    # get test data
    inputs, filters, bias = generate_test_data(np.random,inputs_shape,
        filters_shape)    
    print_info(inputs_shape, filters_shape)
    function_and_names = create_fprop_functions(inputs_shape, filters_shape)
    # try 
    #inputs_2d = np.delete(inputs_2d, [:], 3)
    
    #inputs_2d_theano = shared(np.zeros_like(inputs_2d))
    perf_functions(function_and_names, inputs, filters, bias)
    # init layers
    #get_reference_result
    #for layer in layers: 
    #

    
def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="""Performance experiments for 3d convolution layers.
        Example: ./perf_layers --inputs 15 3 4 5 1 --filters 12 3 4 5 1"""
    )
    parser.add_argument('--inputs', nargs='*', default=[6, 5, 6, 2, 3],
                        help='''Shape of the inputs.''')
    parser.add_argument('--filters', nargs='*', default=[4, 3, 6, 1, 3],
                        help='''Shape of the filters.''')
    args = parser.parse_args()
    # conver to int
    args.inputs = [int(s) for s in args.inputs] 
    args.filters = [int(s) for s in args.filters] 
    return args

if __name__ == '__main__':
    import theano.sandbox.cuda.dnn
    if theano.sandbox.cuda.dnn.dnn_available():
        print("Using cudnn for theano convolutions where they are replaced.")
    else:
        print("Not using cudnn because:\n{:s}".format(
            theano.sandbox.cuda.dnn.dnn_available.msg))
    args = parse_command_line_arguments()
    perf_layers(args.inputs, args.filters)