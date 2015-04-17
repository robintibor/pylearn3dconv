#!/usr/bin/env python
from test_data import generate_test_data
from pylearn2.models.mlp import IdentityConvNonlinearity
import numpy as np
from pylearn2.space import Conv2DSpace
import theano.tensor as T
from pylearn3dconv.layers.cublas_3d_conv import CuBlasConv3dElemwise
from pylearn3dconv.layers.cudnn_3d_conv import CuDnnConv3dElemwise
from pylearn3dconv.layers.theano_3d_2d_conv import Theano3d2dConv3dElemwise
from pylearn3dconv.layers.theano_3d_conv import Theano3dConv3dElemwise
#from pylearn3dconv.layers.cudnn_3d_conv import CuDnn
from pylearn3dconv.volumetric_space import Conv3DSpace
from numpy.random import RandomState
from pylearn3dconv.test import test_function
import sys

def test_layers():
    """ Test layer fprops for different parameter combinations"""
    # Default test
    inputs_shape = [3,3,4,5,3]
    filters_shape = [3,1,4,4,3]
    test_layers_for_parameters(inputs_shape, filters_shape,
        "Default test")
    # All dimensions 1
    inputs_shape = [1,1,1,1,1]
    filters_shape = [1,1,1,1,1]
    test_layers_for_parameters(inputs_shape, filters_shape,
        "Input and filter dimensions 1")
    # Filter spans all dimensions
    # This will lead to a failure for theano 2d3d for some reason
    # (for now we ignore this and remove theano2d3d for this test
    inputs_shape = [3,3,4,5,3]
    filters_shape = [3,3,4,5,3]
    test_layers_for_parameters(inputs_shape, filters_shape,
        "Filter dimension = Input dimension (ignoring theano 3d2d)")
    # Filter smaller for all dimensions
    inputs_shape = [3,3,4,5,3]
    filters_shape = [3,2,2,2,3]
    test_layers_for_parameters(inputs_shape, filters_shape, 
        "Filter dimension < all Input dimension")
    
    # 1,1,1,1,1 filter
    # Filter smaller for all dimensions
    inputs_shape = [3,3,4,5,1]
    filters_shape = [3,1,1,1,1]
    test_layers_for_parameters(inputs_shape, filters_shape, 
        "Filter dimension 1 everywhere")

def test_layers_for_parameters(inputs_shape, filters_shape, testname):
    sys.stdout.write("{:40s} ...".format(testname))
    # Get test data
    rng = RandomState(hash('tobipuma') % 4294967295)
    inputs, filters, bias = generate_test_data(rng, inputs_shape, filters_shape)   
    function_and_names_3d = create_3d_fprops(inputs.shape, filters, bias)
    # (We get different results from theano2d3d 
    # if input time dimension is same as filter time dimension)
    # We want to ignore this, so we take it out 
    if (inputs_shape[3] == filters_shape[3]):
        function_and_names_3d = filter(lambda f: f['name'] != 'Theano 3d2d',
            function_and_names_3d)
    reference_result3d = compute_3d_reference_result(inputs, filters, bias)
    test_functions(function_and_names_3d, inputs, reference_result3d)
    sys.stdout.write(" Ok.\n")

def compute_3d_reference_result(inputs, filters, bias):
    conv3d, _ = create_fprop_layer3d_function(inputs.shape, filters, bias,
        CuBlasConv3dElemwise)
    return conv3d(inputs)

def create_3d_fprops(inputs_shape, filters, bias):
    filters_flipped = filters[:,::-1,::-1,::-1,:]
    fprops = [] 
    fprops.append(compute_3d_func_and_axes('Blas 3d', inputs_shape,
        filters, bias, CuBlasConv3dElemwise))
    fprops.append(compute_3d_func_and_axes('Cudnn 3d', inputs_shape,
        filters, bias, CuDnnConv3dElemwise))
    fprops.append(compute_3d_func_and_axes('Theano 3d', inputs_shape,
        filters, bias, Theano3dConv3dElemwise))
    fprops.append(compute_3d_func_and_axes('Theano 3d2d', inputs_shape,
        filters_flipped, bias, Theano3d2dConv3dElemwise))
    return fprops

def compute_3d_func_and_axes(name, inputs_shape, filters, bias, 
    layer_class):
    function, layer = create_fprop_layer3d_function(inputs_shape, filters, 
        bias, layer_class)
    return({'name': name, 'function': function, 'axes': layer.output_space.axes})

def create_fprop_layer3d_function(inputs_shape, filters, bias, conv_layer_class):
    ftensor5 = T.TensorType('float32', (False,)*5)
    inputs_3d_theano = ftensor5()
    conv_3d_layer = create_layer3d(inputs_shape, filters,
        bias, conv_layer_class)
    conv3d_result = create_fprop_layer_3d(conv_3d_layer, 
        inputs_3d_theano)
    conv3d = theano.function([inputs_3d_theano], conv3d_result)
    return conv3d, conv_3d_layer

def create_fprop_layer_3d_symbolic(inputs_shape, filters,
        bias, conv_layer_class, inputs_3d_theano):
    """ Not used here, just for debugging """
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

def test_functions(function_and_names, inputs, reference_result):
    for function_and_name in function_and_names:
        function = function_and_name['function']
        axes = function_and_name['axes']
        name = function_and_name['name']
        this_reference_result = convert_to_axes(reference_result, axes)
        test_function(function, name, this_reference_result, inputs)
            
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
    
if __name__ == '__main__':
    import theano.sandbox.cuda.dnn
    if theano.sandbox.cuda.dnn.dnn_available():
        print(("Cudnn available, using cudnn for theano convolutions "
            "unless theano optimizer flag set to something else"))
    else:
        print("Not using cudnn because:\n{:s}".format(
            theano.sandbox.cuda.dnn.dnn_available.msg))
    
    test_layers()