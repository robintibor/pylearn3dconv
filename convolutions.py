import numpy as np
import theano
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import theano.tensor.nnet.conv3d2d
conv_mod = SourceModule(open('conv.cu').read())
loop_conv_on_gpu_func = conv_mod.get_function("loop_conv")

def loop_conv_on_gpu(inputs, filters, bias):
    num_batches = inputs.shape[0]
    out_height = inputs.shape[1] - filters.shape[1] + 1;
    out_width = inputs.shape[2] - filters.shape[2] + 1;
    out_duration = inputs.shape[3] - filters.shape[3] + 1;
    num_filters = filters.shape[0]
    output = np.zeros((num_batches, out_height, out_width, out_duration, 
        num_filters)).astype(np.float32)
    loop_conv_on_gpu_func(cuda.In(inputs), cuda.In(filters),
        cuda.In(bias), cuda.InOut(output),
        np.int32(inputs.shape[0]),np.int32(inputs.shape[1]),
        np.int32(inputs.shape[2]),
        np.int32(inputs.shape[3]),np.int32(inputs.shape[4]),
        np.int32(filters.shape[0]),np.int32(filters.shape[1]),
        np.int32(filters.shape[2]), np.int32(filters.shape[3]),
        np.int32(filters.shape[4]),
        block=(1,1,1))
    return output
    
def create_theano_conv3d(inputs, filters, bias):
    conv_theano_result = theano.tensor.nnet.conv3D(V=inputs, W=filters,
                                         b=bias, d=(1,1,1))
    conv_function = theano.function([], conv_theano_result)
    return conv_function

def create_theano_conv3d2d(inputs, filters_flipped, bias):
    # dimshuffles to switch from
    # theano conv3d:   batch x row  x column   x time x channels
    # to 
    # theano conv3d2d: batch x time x channels x row  x column
    conv_result = theano.tensor.nnet.conv3d2d.conv3d(
        signals=inputs.dimshuffle(0,3,4,1,2), 
        filters=filters_flipped.dimshuffle(0,3,4,1,2))
    conv_result = conv_result + bias.dimshuffle('x','x',0,'x','x')
    conv_result = conv_result.dimshuffle(0,3,4,1,2)
    conv_function = theano.function([], conv_result)
    return conv_function

def loop_conv(X, W, b):
    # Go over all five dimensions 
    # (#batches x #height x #width x #dur/length x #channels)
    # with filter that has
    # #filters x #height x #width x #dir # channels
    num_filters = W.shape[0]
    filt_height = W.shape[1]
    filt_width = W.shape[2]
    filt_duration = W.shape[3]
    filt_channels = W.shape[4]
    num_batches = X.shape[0]
    input_height = X.shape[1]
    input_width = X.shape[2]
    input_duration = X.shape[3]
    input_channels = X.shape[4]
    assert(filt_channels == input_channels)
    
    out_height = input_height - filt_height + 1
    out_width = input_width - filt_width + 1
    out_duration = input_duration - filt_duration + 1
    
    # The output is H :)
    H = np.zeros((num_batches,out_height,out_width,out_duration,num_filters))
    for batch_i in xrange(0, num_batches):
        for out_x in xrange(0, out_height):
            for out_y in xrange(0, out_width):
                for out_z in xrange(0, out_duration):
                    for filt_i in xrange(0, num_filters):
                        # Add bias
                        H[batch_i, out_x, out_y, out_z, filt_i] += b[filt_i]
                        for filt_x in xrange(0, filt_height):
                            for filt_y in xrange(0, filt_width):
                                for filt_z in xrange(0, filt_duration):
                                    for filt_chan_i in xrange(0, filt_channels):
                                        weight = W[filt_i, filt_x, filt_y, filt_z, filt_chan_i]
                                        input_val =  X[batch_i, out_x + filt_x, out_y + filt_y, out_z + filt_z, filt_chan_i]
                                        H[batch_i, out_x, out_y, out_z, filt_i] += \
                                             weight * input_val
    return H

def vectorized_conv(X, W, b):
    num_filters = W.shape[0]
    filt_height = W.shape[1]
    filt_width = W.shape[2]
    filt_duration = W.shape[3]
    filt_channels = W.shape[4]
    num_batches = X.shape[0]
    in_height = X.shape[1]
    in_width = X.shape[2]
    in_duration = X.shape[3]
    in_channels = X.shape[4]
    # We assume filter always convolves directly over all channels =>
    # has size of input channels in channel dimension
    assert(filt_channels == in_channels)
    out_height = in_height - filt_height + 1
    out_width = in_width - filt_width + 1
    out_duration = in_duration - filt_duration + 1
    
    # Flatten filters into matrix #filters x #filterpoints
    filter_vec_length = filt_height * filt_width * filt_duration * \
        filt_channels
    filter_mat = np.reshape(W, (num_filters, filter_vec_length))

    # Collect all flattened input into matrix 
    # #filterpoints x #batches*#outputpoints
    flat_out_length = out_height * out_width * out_duration # for single filter/output chan
    inputs_flat = np.zeros((filter_vec_length, num_batches * flat_out_length))
    for out_x in xrange(0, out_height):
        for out_y in xrange(0, out_width):
            for out_z in xrange(0, out_duration):
                # Take input points corresponding
                # to the part of input covered by filters for 
                # given output point(outx, outy, outz)
                # (always take all chans)
                input_part = X[:, out_x:out_x+filt_height, out_y:out_y+filt_width,
                    out_z:out_z+filt_duration, :]
                # Flatten the patches of input
                input_part_flat = \
                    input_part.reshape((num_batches, filter_vec_length)).T
                
                # Now we have columns of flattened input, 
                # each column corresponding to one batch,
                # e.g.for 2 batches and flat filter length of 3
                # column 1 is for input 1 and column 2 is for input 2
                # in1_1 in2_1
                # in1_2 in2_2
                # in1_3 in2_3
                # For easier reshaping we distribute these columns in
                # the final matrix so that all columns for one batch will be
                # adjacent, i.e. in the final matrix, we will have columns:
                # in1 in1 in1 in2 in2 in2
                # instead of:
                # in1 in2 in1 in2 in1 in2
                # flattened out index
                out_index = out_x * out_width * out_duration + out_y * out_duration + out_z
                # Compute inds for that particular output point spaced over whole
                # flattened matrix columns
                # e.g. we can get
                # 0,3 as indices if we have 2 batches and 3 points
                # x - - x - - 
                # and next iteration we get
                # - x - - x - so that all columns of one batch are adjacent
                # as explained above
                inds = np.arange(out_index, 
                    out_index + (num_batches * flat_out_length),
                    step=flat_out_length)
                inputs_flat[:, inds.astype(int)] = input_part_flat
                 
    # Actual computation   
    result_mat = np.dot(filter_mat, inputs_flat)
    # Add bias
    # Also, for the reshape afterwards we first transpose the matrix 
    # (which is #filters x #flatoutputpoints)
    # so that filters are the second dimension and change fastest =>
    # correct H result matrix (note that num_filters is last dimension, not first)
    result_mat = result_mat.T + b
    H = np.reshape(result_mat,  (num_batches, out_height, out_width, out_duration, num_filters))
    return H

def loop_conv_flat(X, W, b):
    # Go over all five dimensions 
    # (#batches x #height x #width x #dur/length x #channels)
    # with filter that has
    # #filters x #height x #width x #dir # channels
    num_filters = W.shape[0]
    filt_height = W.shape[1]
    filt_width = W.shape[2]
    filt_duration = W.shape[3]
    filt_channels = W.shape[4]
    num_batches = X.shape[0]
    input_height = X.shape[1]
    input_width = X.shape[2]
    input_duration = X.shape[3]
    input_channels = X.shape[4]
    assert(filt_channels == input_channels)
    
    out_height = input_height - filt_height + 1
    out_width = input_width - filt_width + 1
    out_duration = input_duration - filt_duration + 1
    
    # The output is H :)
    H = np.zeros((num_batches,out_height,out_width,out_duration,num_filters))
    for batch_i in xrange(0, num_batches):
        for out_x in xrange(0, out_height):
            for out_y in xrange(0, out_width):
                for out_z in xrange(0, out_duration):
                    for filt_i in xrange(0, num_filters):
                        # Add bias
                        H[batch_i, out_x, out_y, out_z, filt_i] += b[filt_i]
                        for filt_x in xrange(0, filt_height):
                            for filt_y in xrange(0, filt_width):
                                for filt_z in xrange(0, filt_duration):
                                    for filt_chan_i in xrange(0, filt_channels):
                                        weight = W[filt_i, filt_x, filt_y, filt_z, filt_chan_i]
                                        input_val =  X[batch_i, out_x + filt_x, out_y + filt_y, out_z + filt_z, filt_chan_i]
                                        H[batch_i, out_x, out_y, out_z, filt_i] += \
                                             weight * input_val
    return H

def dummy_conv(X, W, b):
    # this is where the convolution code should be wrapped
    return np.zeros((16, 15, 21, 13, 10))
