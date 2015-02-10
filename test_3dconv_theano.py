import numpy as np
import theano
#from theano.sandbox.cuda import float32_shared_constructor as shared
from theano import shared
from convolutions import loop_conv
from convolutions import vectorized_conv

def dummy_conv(X, W, b):
    # this is where the convolution code should be wrapped
    return np.zeros((16, 15, 21, 13, 10))

def generate_test_data(rng):
    inputs_shape=(16, 20, 32, 16, 1)
    filters_shape=(10, 6, 12, 4, 1)
    inputs_shape=(10, 12, 17, 13, 1)
    filters_shape=(10, 6, 12, 4, 1)
    inputs_shape=(8, 10, 16, 8, 1)
    filters_shape=(5, 3, 6, 2, 1)
    #inputs_shape=(4, 3, 5, 7, 2)
    #filters_shape=(3, 2, 3, 4, 2)
    #inputs_shape=(3, 2, 4, 5, 2)
    #filters_shape=(1, 2, 3, 2, 2)
    #inputs_shape=(2, 1, 1, 1, 1)
    #filters_shape=(2, 1, 1, 1, 1)
    
    inputs_val = rng.normal(size=inputs_shape).astype('float32')
    filters_val = rng.normal(size=filters_shape).astype('float32')
    
    inputs = shared(inputs_val)
    filters = shared(filters_val)
    bias = shared(np.zeros(filters_shape[0]).astype('float32'))
    bias = shared(rng.normal(size=filters_shape[0]).astype('float32'))
    return inputs, filters, bias

def test_3dconv_theano(rng, conv_fun):

    inputs, filters, bias = generate_test_data(rng)
    ## :note: The GPU implementation is very slow. You are better to use
    ## :func:`conv3d2d <theano.tensor.nnet.conv3d2d.conv3d>` that is faster
    ## on GPU.
    # No flipping needed (from theano source code):
    #    3D "convolution" of multiple filters on a minibatch
    #    (does not flip the kernel, moves kernel with a user specified stride)
    conv_ref = theano.tensor.nnet.conv3D(V=inputs, W=filters,
                                         b=bias, d=(1,1,1))
    f_ref = theano.function([], conv_ref)
    # compute the theano convolution
    res_ref = f_ref()
    print(np.shape(res_ref))
    # compute our convolution
    # res_ours = conv_fun(inputs, filters, bias)
    res_ours = conv_fun(inputs.eval(), filters.eval(), bias.eval())
    
    tol = 1e-4
    distance = np.sum(np.square(res_ref - res_ours))
    if distance > tol:
        print "correct (theano)"
        print res_ref
        print "ours"
        print res_ours
        print 'distance: ' + str(distance)
        assert False
    else:
        print("Result within tolerance, distance {:f}".format(distance))


if __name__ == "__main__":
    test_3dconv_theano(np.random, loop_conv)
    test_3dconv_theano(np.random, vectorized_conv)
