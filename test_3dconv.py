import numpy as np

def dummy_conv(X, W, b):
    # this is where the convolution code should be wrapped
    return np.zeros((X.shape[0], 1, 1, 1, W.shape[0]))

def loop_conv(X, W, b):
    # this is where the convolution code should be wrapped
    # go over all five dimensions 
    # (#batches x #height x #width x #dur/length x #channels)
    # with filter that has
    # #filters x #height x #width x #dir # channels
    num_filters = W.shape[0]
    filter_height = W.shape[1]
    filter_width = W.shape[2]
    filter_duration = W.shape[3]
    filter_channels = W.shape[4]
    num_batches = X.shape[0]
    input_height = X.shape[1]
    input_width = X.shape[2]
    input_duration = X.shape[3]
    input_channels = X.shape[4]
    assert(filter_channels == input_channels)
    
    output_height = input_height - filter_height + 1
    output_width = input_width - filter_width + 1
    output_duration = input_duration - filter_duration + 1
    
    
    # The output is H :)
    H = np.zeros((num_batches,output_height,output_width,output_duration,num_filters))
    for batch_i in xrange(0, num_batches):
        for out_x in xrange(0, output_height):
            for out_y in xrange(0, output_height):
                for out_z in xrange(0, output_duration):
                    for filt_i in xrange(0, num_filters):
                        H[batch_i, out_x, out_y, out_z, filt_i] += b[filt_i]
                        for filt_x in xrange(0, filter_height):
                            for filt_y in xrange(0, filter_width):
                                for filt_z in xrange(0, filter_duration):
                                    for filt_chan_i in xrange(0, filter_channels):
                                        weight = W[filt_i, filt_x, filt_y, filt_z, filt_chan_i]
                                        input_val =  X[batch_i, out_x + filt_x, out_y + filt_y, out_z + filt_z, filt_chan_i]
                                        H[batch_i, out_x, out_y, out_z, filt_i] += \
                                             weight * input_val
                                        
    return H

def vectorized_conv(X, W, b):
    #... first make filters into vectors
    # ... then make input image into several vectors that need to get multiplied..
    # in the end reshape back ...

def test_3dconv_img_filter(rng, conv_fun):        
    # so we have 1 dimension for the batch
    batchSize = rng.randint(1, 10)
    # and 3 dimensions for the actual convolution (2 spatial 1 temporal)
    duration = rng.randint(3, 10)
    width = rng.randint(1, 5)
    height = rng.randint(1, 5)
    # and this is the number of input values (e.g. 3 in the case of RGB)
    inputChannels = rng.randint(1, 4)

    # for this first simple test we are going to assume that the filter
    # is as big as the image which reduces convolution to the fully connected case
    filterWidth = width
    filterHeight = height
    filterDur = duration
    numFilters = rng.randint(1, 3)

    # setup weights and biases
    W = rng.randn(numFilters, filterHeight, filterWidth, filterDur, inputChannels)
    b = rng.randn(numFilters)

    # setup input
    X = rng.randn(batchSize, height, width, duration, inputChannels)

    # compute the hidden representation after convolution using the function
    H_computed = conv_fun(X, W, b)
    assert H_computed.shape[1] == 1
    assert H_computed.shape[2] == 1
    assert H_computed.shape[3] == 1

    # compute the convolution as matrix multiplication
    n = inputChannels * height * width * duration
    W_mat = np.zeros((n, numFilters))
    X_mat = np.zeros((batchSize, n))
    H_computed_mat = np.zeros((batchSize, numFilters))

    for qi in xrange(0, numFilters):
        W_mat[:, qi] = np.reshape(W[qi, :, :, :, :], (n))
        H_computed_mat[:, qi] = H_computed[:, 0, 0, 0, qi]
    for qi in xrange(0, batchSize):
        X_mat[qi, :] = np.reshape(X[qi, :, :, :, :], (n))

    # compute the hidden representation after convolution using matrix multiplication
    H_mat = np.dot(X_mat, W_mat) + b

    tol = 1e-4

    max_err = np.abs(H_computed_mat - H_mat).max()
    if max_err  > tol:
        print H_computed_mat
        print H_mat
        print 'max error: ' + str(max_err)
        assert False


if __name__ == "__main__":
    #test_3dconv_img_filter(np.random, dummy_conv)
    test_3dconv_img_filter(np.random, loop_conv)
