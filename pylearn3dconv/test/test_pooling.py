import numpy as np


def max_pool_3d_numpy(inputs, pool_shape, pool_stride):
    assert (len(inputs.shape) == 5)
    assert(len(pool_shape) == 3)
    assert(len(pool_stride) == 3)
    # in b c 0 1 2 format
    output_shape = [((inputs.shape[i+2] - pool_shape[i]) // pool_stride[i]) + 1 
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
    pass
    # create inputs for pooling
    # get result numpy and dnn, compare