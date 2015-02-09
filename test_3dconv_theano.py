import numpy as np
import theano
from theano.sandbox.cuda import float32_shared_constructor as shared

def dummy_conv(X, W, b):
    # this is where the convolution code should be wrapped
    return np.zeros((16, 15, 21, 13, 10))


def test_3dconv_theano(rng, conv_fun):
    inputs_shape=(16, 20, 32, 16, 1)
    filters_shape=(10, 6, 12, 4, 1)
    
    inputs_val = rng.normal(size=inputs_shape).astype('float32')
    filters_val = rng.normal(size=filters_shape).astype('float32')
    
    inputs = shared(inputs_val)
    filters = shared(filters_val)
    bias = shared(np.zeros(filters_shape[0]).astype('float32'))

    # we might have to flip filters here, not sure ...
    filters_theano = filters[:,::-1,::-1,::-1,:]
    ## :note: The GPU implementation is very slow. You are better to use
    ## :func:`conv3d2d <theano.tensor.nnet.conv3d2d.conv3d>` that is faster
    ## on GPU.
    conv_ref = theano.tensor.nnet.conv3D(V=inputs, W=filters_theano,
                                         b=bias, d=(1,1,1))
    f_ref = theano.function([], conv_ref)
    # compute the theano convolution
    res_ref = f_ref()
    print(np.shape(res_ref))
    # compute our convolution
    res_ours = conv_fun(inputs, filters, bias)
    
    tol = 1e-4
    distance = np.sum(np.square(res_ref - res_ours))
    if distance > tol:
        print res_ref
        print res_ours
        print 'distance: ' + str(distance)
        assert False


if __name__ == "__main__":
    test_3dconv_theano(np.random, dummy_conv)
