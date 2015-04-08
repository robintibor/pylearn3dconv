import theano
# also see http://deeplearning.net/software/theano/tutorial/gpu_data_convert.html 
# for pycuda /theano interplay
import theano.misc.pycuda_init
from test_data import generate_theano_test_data
from convolutions import loop_conv_on_gpu, create_theano_conv3d, \
    vectorized_conv, loop_conv, create_theano_conv3d2d

from timeit import default_timer as timer
import numpy as np
import argparse

def perffunc(name, func, correct_result, *func_args):
    total_running_time = 0
    runs = 0
    while total_running_time < 1:
        start = timer()
        if (len(func_args) > 0):
            result = func(*func_args)
        else:
            result = func()
        end = timer()
        total_running_time += end - start
        runs += 1
    print("{:25s} runtime per iteration ({:4d} iterations): {:8.4f}ms".format(
        name, runs, 1000 * total_running_time / runs))
    diff = np.sum(np.square(correct_result - result))
    assert diff < 1e-5

def perf_3d_convs(inputs_shape, filters_shape):
    #inputs_shape=(15, 20, 16, 15, 3)
    #filters_shape=(7, 3, 6, 2, 3)
    inputs, filters, bias = generate_theano_test_data(np.random,inputs_shape,
        filters_shape)
    print("Batches/Filters, rows, columns, times, channels")
    print("Input shape  {:s}".format(inputs_shape))
    print("Filter shape {:s}".format(filters_shape))
    print("#Inputs  {:7d}".format(np.prod(inputs_shape)))
    print("#Weights {:7d}".format(np.prod(filters_shape)))
    theano_3d_func = create_theano_conv3d(inputs, filters, bias)
    # flipping from https://groups.google.com/forum/#!msg/theano-users/1S9_bZgHxVw/0cQR9a4riFUJ
    filters_flip = filters[:,::-1,::-1,::-1,:]  # flip width, height and time
    theano_3d_2d = create_theano_conv3d2d(inputs, filters_flip, bias)
    inputs_val =  np.array(inputs.eval())
    filters_val = np.array(filters.eval())
    bias_val = np.array(bias.eval())
    correct_result = theano_3d_func()
    perffunc("Theano 3d", theano_3d_func, correct_result)
    perffunc("Theano 3d2d", theano_3d_2d, correct_result)
    perffunc("GPU Loop Conv", loop_conv_on_gpu, correct_result, 
        inputs_val, filters_val, bias_val)
    perffunc("Python Vectorized Conv", vectorized_conv, correct_result, 
        inputs_val, filters_val, bias_val)
    perffunc("Python Loop Conv", loop_conv, correct_result, 
        inputs_val, filters_val, bias_val)
    
def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="""Performance experiments for 3d convolutions.
        Example: ./perf_convs --inputs 15 3 4 5 1 --filters 12 3 4 5 1"""
    )
    parser.add_argument('--inputs', nargs='*', default=[15, 20, 16, 15, 3],
                        help='''Shape of the inputs.''')
    parser.add_argument('--filters', nargs='*', default=[7, 3, 6, 2, 3],
                        help='''Shape of the filters.''')
    args = parser.parse_args()
    # conver to int
    args.inputs = [int(s) for s in args.inputs] 
    args.filters = [int(s) for s in args.filters] 
    return args

if __name__ == "__main__":
    args = parse_command_line_arguments()
    perf_3d_convs(args.inputs, args.filters)
