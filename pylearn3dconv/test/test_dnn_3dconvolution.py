#!/usr/bin/env python
from pylearn3dconv.test import generate_test_data
import numpy as np
import theano.tensor as T
from pylearn3dconv.theanodnn.conv import dnn_3dconv
from numpy.random import RandomState
from pylearn3dconv.test import test_function
import sys

def test_convolution():
    """ Test 3d convolutions for different parameter combinations"""
    # Default test
    inputs_shape = [3,3,4,5,3]
    filters_shape = [3,1,4,4,3]
    test_convolution_for_parameters(inputs_shape, filters_shape,
        "Default test")
    # All dimensions 1
    inputs_shape = [1,1,1,1,1]
    filters_shape = [1,1,1,1,1]
    test_convolution_for_parameters(inputs_shape, filters_shape,
        "Input and filter dimensions 1")
    # Filter spans all dimensions
    # This will lead to a failure for theano 2d3d for some reason
    # (for now we ignore this and remove theano2d3d for this test
    inputs_shape = [3,3,4,5,3]
    filters_shape = [3,3,4,5,3]
    test_convolution_for_parameters(inputs_shape, filters_shape,
        "Filter dimension = Input dimension")
    # Filter smaller for all dimensions
    inputs_shape = [3,3,4,5,3]
    filters_shape = [3,2,2,2,3]
    test_convolution_for_parameters(inputs_shape, filters_shape, 
        "Filter dimension < all Input dimension")
    # 1,1,1,1,1 filter
    # Filter smaller for all dimensions
    inputs_shape = [3,3,4,5,1]
    filters_shape = [3,1,1,1,1]
    test_convolution_for_parameters(inputs_shape, filters_shape, 
        "Filter dimension 1 everywhere")

def test_convolution_for_parameters(inputs_shape, filters_shape, testname):
    sys.stdout.write("{:40s} ... ".format(testname))
    # Get test data
    rng = RandomState(hash('tobipuma') % 4294967295)
    inputs, filters = generate_conv_test_data(rng, inputs_shape, filters_shape)
    reference_result = compute_reference_result(inputs, filters)
    dnn_3d_func = create_cudnn_3d_conv()
    test_function(dnn_3d_func, testname, reference_result, inputs, filters)
    sys.stdout.write(" Ok.\n")

def generate_conv_test_data(rng, inputs_shape, filters_shape):
    """ Have to transpose inputs and filters from b012c to bc012 """
    inputs, filters, _ = generate_test_data(rng, inputs_shape, filters_shape)
    inputs_dnn = inputs.transpose(0,4,1,2,3)
    filters_dnn = filters.transpose(0,4,1,2,3)
    return inputs_dnn, filters_dnn
    
def create_cudnn_3d_conv():
    ftensor5 = T.TensorType('float32', (False,)*5)
    inputs = ftensor5()
    filters = ftensor5()
    kernel_stride = (1,1,1)
    result = dnn_3dconv(inputs, filters, subsample=kernel_stride,
            conv_mode='cross')
    return theano.function([inputs, filters], result)

def compute_reference_result(inputs, filters):
    return loop_conv(inputs, filters)

def loop_conv(X, W):
    """ Gold standard convolution for test, looping over all dimensions.
    Actually performing cross correlation, not convolution. 
    Assumes bc012 input"""
    # Go over all five dimensions 
    # (#batches x #channels x #height x #width x #dur/length )
    # with filter that has
    # #filters x #channels x #height x #width x #dur/length 
    num_filters = W.shape[0]
    filt_channels = W.shape[1]
    filt_height = W.shape[2]
    filt_width = W.shape[3]
    filt_duration = W.shape[4]
    num_batches = X.shape[0]
    input_channels = X.shape[1]
    assert(filt_channels == input_channels)
    out_shape = compute_out_shape(X.shape, W.shape)
    out_height = out_shape[2]
    out_width = out_shape[3]
    out_duration = out_shape[4]
    
    # The output is H :)
    H = np.zeros((out_shape))
    for batch_i in xrange(0, num_batches):
        for filt_i in xrange(0, num_filters):
            for out_x in xrange(0, out_height):
                for out_y in xrange(0, out_width):
                    for out_z in xrange(0, out_duration):
                        for chan_i in xrange(0, filt_channels):
                            for filt_x in xrange(0, filt_height):
                                for filt_y in xrange(0, filt_width):
                                    for filt_z in xrange(0, filt_duration):
                                        weight = W[filt_i, chan_i, filt_x, filt_y, filt_z]
                                        input_val =  X[batch_i, chan_i, \
                                            out_x + filt_x, out_y + filt_y, out_z + filt_z]
                                        H[batch_i, filt_i, out_x, out_y, out_z] += \
                                             weight * input_val
    return H

def compute_out_shape(inputs_shape, filters_shape):
    num_batches = inputs_shape[0]
    out_height = inputs_shape[2] - filters_shape[2] + 1;
    out_width = inputs_shape[3] - filters_shape[3] + 1;
    out_duration = inputs_shape[4] - filters_shape[4] + 1;
    num_filters = filters_shape[0]
    return (num_batches, num_filters, out_height, out_width, out_duration)
    
    
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
    
    test_convolution()
