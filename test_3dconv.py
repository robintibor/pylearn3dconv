import numpy as np

def dummy_conv(X, W, b):
    # this is where the convolution code should be wrapped
    return np.zeros((X.shape[0], 1, 1, 1, W.shape[0]))


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
    test_3dconv_img_filter(np.random, dummy_conv)
