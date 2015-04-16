from vol_conv.test_data import generate_test_data
from vol_conv.layers.cudnn_3d_conv import CuDnnConv3dElemwise
from vol_conv.layers.cublas_3d_conv import CuBlasConv3dElemwise
import argparse
from vol_conv.perf_layers import create_fprop_layer3d_function 
from numpy.random import RandomState
import numpy as np
from vol_conv.perf import perf_func
from vol_conv.volumetric_space import Conv3DSpace
import gc
from theano import shared

def perf_single_layer(layer_class, layername, inputs_shape, filters_shape,
    frames_min, frames_max):
    rng = RandomState(np.uint32(hash('tobipuma')))
    average_runtimes = []
    for frames in xrange(frames_min, frames_max + 1):
        gc.collect() # force collection to free gpu memory for initializing
        # next layer (!)
        inputs_shape[3] = frames
        runtime, iterations = perf_single_layer_for_input_shape(layer_class, 
            layername, inputs_shape, filters_shape, rng)
        average_runtimes.append(runtime/iterations)
        print("For {:2d} frames: {:5.2f}ms".format(frames,
            (runtime * 1000.0) / (iterations)))

def perf_single_layer_for_input_shape(layer_class, layername, inputs_shape,
    filters_shape, rng):
    inputs, filters, bias = generate_test_data(rng, inputs_shape, filters_shape)
    fprop_function, _ = create_fprop_layer3d_function(inputs_shape, filters, 
        bias, layer_class)
    runtime, iterations = perf_func(fprop_function, None, inputs)
    return runtime, iterations

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="""Performance experiments for 3d convolution layers.
        Example: perf/perf_single_3d_layer.py --inputs 15 3 4 5 1 
        --filters 12 3 4 5 1 --maxframes 30 --layer cudnn"""
    )
    parser.add_argument('--inputs', nargs='*', default=[4, 3, 2, 3],
                        dest='inputs_shape',
                        help='''Shape of the inputs b 0 1 c format.
                        Time dimension will be determined by timemax parameter''')
    parser.add_argument('--filters', nargs='*', default=[3, 2, 2, 1, 3],
                        dest='filters_shape',
                        help='''Shape of the filters in b 0 1 2 c format.''')
    parser.add_argument('--minframes',  default=None, type=int,
                        dest='frames_min',
                        help='''Maximum number of frames/3rd dimension to try.
                        Will test performance from this frame number to max frame number.''')
    parser.add_argument('--maxframes',  default=20, type=int,
                        dest='frames_max',
                        help='''Maximum number of frames/3rd dimension to try.
                        Will test performance until this number of frames.''')
    parser.add_argument('--layer', default='cudnn', dest='layername',
                        choices=_layername_to_class_dict.keys(),
                        help='''Layer to profile.''')
    args = parser.parse_args()
    # conver to int
    args.inputs_shape = [int(s) for s in args.inputs_shape]
    # add time dimension, just set to 1 for now
    args.inputs_shape = args.inputs_shape[0:3] + [1] + [args.inputs_shape[3]]
    args.filters_shape = [int(s) for s in args.filters_shape] 
    if (args.frames_min is None):
        # need minimum as many input frames as filter frames
        args.frames_min = args.filters_shape[3]
    args.layer_class = layername_to_class(args.layername)
    return args

_layername_to_class_dict = {
        'cudnn': CuDnnConv3dElemwise,
        'cublas': CuBlasConv3dElemwise,
        }


def layername_to_class(name):
    return _layername_to_class_dict[name]

if __name__ == '__main__':
    args = parse_command_line_arguments()
    perf_single_layer(args.layer_class, args.layername, args.inputs_shape, 
        args.filters_shape, args.frames_min, args.frames_max)
    # python perf/perf_single_3d_layer.py --inputs 15 3 4 5 3 --filters 12 3 4 5 3 --maxframes 30 --layer cudnn
    # python perf/perf_single_3d_layer.py --inputs 32 80 80 3 --filters 32 5 5 5 3 --maxframes 30 --layer cudnn
    # python perf/perf_single_3d_layer.py --inputs 57 80 80 3 --filters 64 5 5 5 3 --maxframes 30 --layer cudnn