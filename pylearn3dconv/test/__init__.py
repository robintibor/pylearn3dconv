import numpy as np
import theano.tensor as T
from theano import shared

def test_function(function, name, reference_result, *func_args):
    if (len(func_args) > 0):
            result = function(*func_args)
    else:
        result = function()
    if (not np.allclose(result, reference_result,
        atol=1e-3, rtol=0)):
        diff = np.sum(np.abs(result - reference_result))
        sum_of_squared_diff = np.sum(np.square(result - reference_result))
        raise AssertionError("Failure for function {:s},\n"
            "Sum of absolute diff to reference result {:f}\n"
            "Sum of squared diff to refence_result {:f}".format(name,
            diff, sum_of_squared_diff))
        
ftensor5 = T.TensorType('float32', (False,)*5)

def generate_theano_test_data(rng,inputs_shape=(8, 10, 16, 8, 1),
        filters_shape=(5, 3, 6, 2, 1)):
    inputs_val,filters_val, bias_val = generate_test_data(rng,
        inputs_shape, filters_shape)
    inputs = shared(inputs_val)
    filters = shared(filters_val)
    bias = shared(bias_val)
    return inputs, filters, bias

def generate_test_data(rng, inputs_shape=(8, 10, 16, 8, 1),
        filters_shape=(5, 3, 6, 2, 1)):
    inputs = rng.normal(size=inputs_shape).astype('float32')
    filters = rng.normal(size=filters_shape).astype('float32')
    bias = rng.normal(size=filters_shape[0]).astype('float32')
    return inputs, filters, bias