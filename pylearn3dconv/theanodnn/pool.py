import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_pool

def dnn_pool3d2d(inputs, pool_shape, pool_stride, image_shape, mode='max'):
    for i in xrange(3):
        assert pool_shape[i] <= image_shape[i], ("pool shape should be less"
            " or equal than image shape, {:d} > {:d} for "
            "pool_shape: {:s}, image_shape:{:s}").format(pool_shape[i],
                image_shape[i], pool_shape, image_shape)
    first_2d_pooled_outputs = []
    for z in range(image_shape[2]):
        pooled_slice = dnn_pool(inputs[:,:,:,:,z], ws=pool_shape[0:2], 
            stride=pool_stride[0:2], mode=mode)
        first_2d_pooled_outputs.append(pooled_slice)
    
    first_2d_pooled_output = T.stack(first_2d_pooled_outputs)[0,:,:,:,:,:]
    first_2d_pooled_output = first_2d_pooled_output.dimshuffle(1,2,3,4,0)
    # now 1d-pool over last dimension...
    # could use first or second dimension as input of pool1d..
    # compute maximum y index after first pooling
    max_y = ((image_shape[1] - pool_shape[1]) // pool_stride[1]) + 1
    final_outputs = []
    for y in range(max_y):
        final_pooled_slice = dnn_pool(first_2d_pooled_output[:,:,:,y,:], 
            ws=(1, pool_shape[2]), 
            stride=(1, pool_stride[2]), mode=mode)
        final_outputs.append(final_pooled_slice)
    
    final_output = T.stack(final_outputs)[0,:,:,:,:,:]     
    final_output = final_output.dimshuffle(1,2,3,0,4)
    return final_output