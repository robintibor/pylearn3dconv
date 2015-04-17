import numpy as np
from numpy.random import RandomState
from pylearn3dconv.theanodnn.pool import dnn_pool3d2d
from pylearn3dconv.test import  ftensor5, test_function
import theano
import sys

def test_pooling():
    inputs_shape = [3,2,3,4,2]
    pool_shape = (2,1,2)
    pool_stride = (1,2,1)
    test_pooling_for_parameters(inputs_shape, pool_shape, pool_stride,
        "Default test")
    inputs_shape = [3,2,3,4,2]
    pool_shape = (3,4,2)
    pool_stride = (1,1,1)
    test_pooling_for_parameters(inputs_shape, pool_shape, pool_stride,
        "Pooling shape == Image shape")
    inputs_shape = [3,2,3,4,2]
    pool_shape = (2,2,1)
    pool_stride = (5,5,5)
    test_pooling_for_parameters(inputs_shape, pool_shape, pool_stride,
        "Strides larger than input")

def test_pooling_for_parameters(inputs_shape, pool_shape, pool_stride,
        testname):
    sys.stdout.write("{:40s} ...".format(testname))
    rng = RandomState(hash('tobipuma') % 4294967295) 
    inputs = rng.normal(size=inputs_shape).astype(np.float32)
    dnn_pool3d2d_func = create_dnnpool3d2d_func(pool_shape, pool_stride,
        inputs_shape[2:])
    reference_result = max_pool_3d_numpy(inputs, pool_shape, pool_stride)
    test_function(dnn_pool3d2d_func, testname, reference_result, inputs)
    sys.stdout.write(" Ok.\n")

def create_dnnpool3d2d_func(pool_shape, pool_stride, image_shape):
    inputs_theano = ftensor5()
    result = dnn_pool3d2d(inputs_theano, pool_shape, pool_stride, image_shape)
    return theano.function([inputs_theano], result)
    
def max_pool_3d_numpy(inputs, pool_shape, pool_stride):
    assert (len(inputs.shape) == 5) # for now restrict to 3d images
    assert(len(pool_shape) == 3)
    assert(len(pool_stride) == 3)
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


if __name__ == "__main__":
    test_pooling()