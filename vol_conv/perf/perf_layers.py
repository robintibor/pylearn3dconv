#!/usr/bin/env python
from vol_conv.test_data import generate_test_data
from pylearn2.models.mlp import ConvElemwise, IdentityConvNonlinearity
""" Transform this into test instead!!!"""
import theano
import theano.misc.pycuda_init
from vol_conv.convolutions import  compute_out_shape, vectorized_conv
import numpy as np
import argparse
from pylearn2.space import Conv2DSpace
import theano.tensor as T
from perf import perf_func_print_results
from vol_conv.layers.blas2d_manuel_conv import ConvElemwiseBlas
from vol_conv.layers.cublas_3d_conv import CuBlasConv3dElemwise
from vol_conv.layers.cudnn_3d_conv import CuDnnConv3dElemwise
from vol_conv.layers.theano_3d_2d_conv import Theano3d2dConv3dElemwise
from vol_conv.layers.theano_3d_conv import Theano3dConv3dElemwise
#from vol_conv.layers.cudnn_3d_conv import CuDnn
from vol_conv.volumetric_space import Conv3DSpace
from numpy.random import RandomState
import theano.sandbox.cuda
import gc
from theano.sandbox.cuda.basic_ops import gpu_from_host, gpu_contiguous
# 5 dimensional tensor type for 3d convolutions:
ftensor5 = T.TensorType('float32', (False,)*5)

def perf_layers(inputs_shape, filters_shape):
    # get test data
    rng = RandomState(hash('tobipuma') % 4294967295)
    inputs, filters, bias, inputs_2d, filters_2d = generate_2d_3d_test_data(
        rng, inputs_shape, filters_shape)   
    print_info(inputs_shape, filters_shape)
    reference_result2d = compute_2d_reference_result(inputs_2d,filters_2d,bias)
    reference_result3d = compute_3d_reference_result(inputs, filters, bias)
    function_and_names_2d = create_2d_fprops(inputs_2d.shape, filters_2d, bias)
    function_and_names_3d = create_3d_fprops(inputs.shape, filters, bias)
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

def compute_2d_reference_result(inputs, filters, bias):
    conv2d, _ = create_fprop_layer2d_function(inputs.shape, np.copy(filters), 
        np.copy(bias),
        ConvElemwise)
    conv_2d_result = conv2d(inputs)
    conv_2d_result_np = np.array(conv_2d_result)
    del conv2d
    del conv_2d_result
    return conv_2d_result_np

def compute_3d_reference_result(inputs, filters, bias):
    conv3d, _ = create_fprop_layer3d_function(inputs.shape, filters, bias,
        CuBlasConv3dElemwise)
    return conv3d(inputs)

def create_2d_fprops(inputs_shape, filters, bias):
    filters_flipped = filters[:,::-1,::-1,:]
    yield compute_2d_func_and_axes('conv2d', inputs_shape, filters, 
        bias, ConvElemwise)
    yield compute_2d_func_and_axes('conv2dblas', inputs_shape, 
        filters_flipped, bias, ConvElemwiseBlas)

def create_3d_fprops(inputs_shape, filters, bias):
    filters_flipped = filters[:,::-1,::-1,::-1,:]    
    yield compute_3d_func_and_axes('Blas 3d', inputs_shape,
        filters, bias, CuBlasConv3dElemwise)
    yield compute_3d_func_and_axes('Cudnn 3d', inputs_shape,
        filters, bias, CuDnnConv3dElemwise)
    yield compute_3d_func_and_axes('Theano 3d', inputs_shape,
        filters, bias, Theano3dConv3dElemwise)
    # This last as it will fail first for bigger filters
    yield compute_3d_func_and_axes('Theano 3d2d', inputs_shape,
        filters_flipped, bias, Theano3d2dConv3dElemwise)

def compute_2d_func_and_axes(name, inputs_shape, filters, bias, 
    layer_class):
    function, layer = create_fprop_layer2d_function(inputs_shape, filters, 
        bias, layer_class)
    return {'name': name, 'function': function, 'axes': layer.output_space.axes}

def compute_3d_func_and_axes(name, inputs_shape, filters, bias, 
    layer_class):
    function, layer = create_fprop_layer3d_function(inputs_shape, filters, 
        bias, layer_class)
    return({'name': name, 'function': function, 'axes': layer.output_space.axes})


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
    # keep variable on gpu for perfing
    if (theano.config.device.startswith('gpu')):
        conv2d_result = gpu_from_host(conv2d_result)
    conv2d = theano.function([inputs_2d_theano], conv2d_result)
    return conv2d, conv_2d_layer

def create_fprop_layer3d_function(inputs_shape, filters, bias, conv_layer_class):
    inputs_3d_theano = ftensor5()
    conv_3d_layer = create_layer3d(inputs_shape, filters,
        bias, conv_layer_class)
    conv3d_result = create_fprop_layer_3d(conv_3d_layer, 
        inputs_3d_theano)
    conv3d = theano.function([inputs_3d_theano], conv3d_result)
    return conv3d, conv_3d_layer

def create_fprop_layer_3d_symbolic(inputs_shape, filters,
        bias, conv_layer_class, inputs_3d_theano):
    conv_3d_layer = create_layer3d(inputs_shape, filters,
        bias, conv_layer_class)
    conv3d_result = create_fprop_layer_3d(conv_3d_layer, 
        inputs_3d_theano)
    return conv3d_result

def create_layer3d(inputs_shape, filters, bias, conv_layer_class):
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
    # convert filters to correct axes (('b', 0, 1, 2, ' c') are test data axes)
    converted_filters = Conv3DSpace.convert_numpy(filters, 
        ('b', 0, 1, 2, 'c'), conv_3d_layer.detector_space.axes)
    conv_3d_layer.set_weights(converted_filters)
    conv_3d_layer.set_biases(bias)
    return conv_3d_layer
 
def create_fprop_layer_3d(conv_3d_layer, inputs_3d_theano):
    conv3d_result = conv_3d_layer.fprop(inputs_3d_theano)
    # keep variable on gpu for perfing
    # Lets remove for now, we will perf differently anyways most likely
    #if (theano.config.device.startswith('gpu')):
    #    conv3d_result = gpu_from_host(conv3d_result)
    return conv3d_result

def compute_first_result(function_and_names, inputs):
    """ Convenience function for computing a reference result for all tests. """
    function = function_and_names[0]['function']
    result = function(inputs)
    return result

def perf_functions(function_and_names, inputs, reference_result):
    for function_and_name in function_and_names:
        name = function_and_name['name']
        function = function_and_name['function']
        axes = function_and_name['axes']
        this_reference_result = convert_to_axes(reference_result, axes)
        perf_func_print_results(name, function, this_reference_result, inputs)
        del function
        gc.collect() # clear shared memory

def convert_to_axes(reference_result, axes):
    # assuming we have b c 0 1 (2) for reference
    if (reference_result.ndim == 4):
        return Conv2DSpace.convert_numpy(reference_result, 
            ('b', 'c', 0, 1), axes)
    elif (reference_result.ndim == 5):
        return Conv3DSpace.convert_numpy(reference_result, 
            ('b', 'c', 0, 1, 2), axes)
    else:
        raise ValueError(("Expect result to have 4 or 5 dims, " 
            "has {:d} dims".format(reference_result.ndim)))

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
    parser.add_argument('--inputs', nargs='*', default=[20, 35, 16, 20, 3],
                        help='''Shape of the inputs.''')
    parser.add_argument('--filters', nargs='*', default=[5, 14, 6, 3, 3],
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
    # failure at: perf/perf_layers.py --inputs 57 80 80 20 3 --filters 64 5 5 5 3
    #perf/perf_layers.py --inputs 15 80 80 15 3 --filters 12 5 5 5 3
    
    #perf/perf_layers.py --inputs 32 80 80 15 3 --filters 32 5 5 5 3