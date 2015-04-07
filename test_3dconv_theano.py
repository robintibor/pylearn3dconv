import numpy as np
import theano
#from theano.sandbox.cuda import float32_shared_constructor as shared
from theano import shared
from convolutions import loop_conv
from convolutions import vectorized_conv
from timeit import default_timer as timer
import theano.tensor.nnet.conv3d2d
from test_data import generate__theano_test_data

def test_3dconv_theano(rng, conv_fun):

    inputs, filters, bias = generate__theano_test_data(rng)
    ## :note: The GPU implementation is very slow. You are better to use
    ## :func:`conv3d2d <theano.tensor.nnet.conv3d2d.conv3d>` that is faster
    ## on GPU.
    # No flipping needed (from theano source code):
    #    3D "convolution" of multiple filters on a minibatch
    #    (does not flip the kernel, moves kernel with a user specified stride)
   
    # flipping from https://groups.google.com/forum/#!msg/theano-users/1S9_bZgHxVw/0cQR9a4riFUJ
    #filters_flip = filters[:,::-1,:,::-1,::-1]  # flip time, width and height
    #conv_ref = theano.tensor.nnet.conv3d2d.conv3d(signals=inputs, 
    #                                    filters=filters_flip)
    #conv_ref = conv_ref + bias.dimshuffle('x','x',0,'x','x')
    
    conv_ref = theano.tensor.nnet.conv3D(V=inputs, W=filters,
                                         b=bias, d=(1,1,1))
    
    f_ref = theano.function([], conv_ref)
    # compute the theano convolution
    start = timer()
    res_ref = f_ref() 
    end = timer()
    print("Theano: {:7.4f}ms".format((end - start) * 1000))
    print(np.shape(res_ref))
    # compute our convolution
    # res_ours = conv_fun(inputs, filters, bias)
    start = timer()
    res_ours = conv_fun(np.array(inputs.eval()),np.array(filters.eval()), 
        np.array(bias.eval()))
    end = timer()
    print("Ours: {:7.4f}ms".format((end - start) * 1000))

    tol = 1e-4
    distance = np.sum(np.square(res_ref - res_ours))
    if distance > tol:
        print "correct (theano)"
        print res_ref
        print "ours"
        print res_ours
        print 'distance: ' + str(distance)
        assert False, "distance greater than tolerance"
    else:
        print("Result within tolerance, distance {:f}".format(distance))

if __name__ == "__main__":
    test_3dconv_theano(np.random, loop_conv)
    test_3dconv_theano(np.random, vectorized_conv)
