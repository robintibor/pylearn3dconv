import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_pool
from theano.sandbox.cuda.basic_ops import gpu_alloc_empty

def dnn_pool3d2d(inputs, pool_shape, pool_stride, image_shape, mode='max'):
    """ Pool first all time-slices, so 2d-poolings over width and height.
    Then do a 1dpooling over the time (done as fake2d pooling with pooling shape
    1 for the ignored dimension."""
    for i in xrange(3):
        assert pool_shape[i] <= image_shape[i], ("pool shape should be less"
            " or equal than image shape, {:d} > {:d} for "
            "pool_shape: {:s}, image_shape:{:s}").format(pool_shape[i],
                image_shape[i], pool_shape, image_shape)
    output_shape = [((image_shape[i] - pool_shape[i]) // pool_stride[i]) + 1 
        for i in xrange(3)]
    output2d_pooled = gpu_alloc_empty(inputs.shape[0], inputs.shape[1],
        output_shape[0], output_shape[1], image_shape[2])
    for z in range(image_shape[2]):
        pooled_slice = dnn_pool(inputs[:,:,:,:,z], ws=pool_shape[0:2], 
            stride=pool_stride[0:2], mode=mode)
        output2d_pooled = T.set_subtensor(output2d_pooled[:,:,:,:,z], pooled_slice)
    
    
    # now 1d-pool over last dimension...
    # could use first or second dimension as input of pool1d..
    # compute maximum y index after first pooling
    output = gpu_alloc_empty(inputs.shape[0], inputs.shape[1],
        output_shape[0], output_shape[1], output_shape[2])
    max_y = output_shape[1]
    for y in range(max_y):
        # ignore first=0 dimension, alrdy pooled in loop before
        # so set stride and shape to 1 there
        final_pooled_slice = dnn_pool(output2d_pooled[:,:,:,y,:], 
            ws=(1, pool_shape[2]), 
            stride=(1, pool_stride[2]), mode=mode)
        output = T.set_subtensor(output[:,:,:,y,:], final_pooled_slice)

    return output
    