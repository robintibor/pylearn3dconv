from pylearn3dconv.test import ftensor5
from numpy.random import RandomState
import numpy as np
from pylearn3dconv.volumetric_space import Conv3DSpace
from pylearn3dconv.layers.variants import CuDnnConv3dElemwise,\
    CuBlasConv3dElemwise
from pylearn2.models.mlp import Softmax, MLP, IdentityConvNonlinearity
import theano.tensor as T
from pylearn3dconv.perf import perf_func_print_results
import theano
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.config import yaml_parse
import os
import argparse

def perf_mlp2d(inputs_shape, layer_class, modelname):
    rng = RandomState(np.uint32(hash('perfthemlp')))
    # generate mlp grad
    # generate inputs
    # perf....
    inputs = rng.normal(size=inputs_shape).astype(np.float32)
    mlp_grad_func = create_mlp_grad_func(inputs_shape, layer_class, modelname)
    mlp_grad_func(inputs)
    perf_func_print_results(modelname, mlp_grad_func, None, inputs)

def create_mlp_grad_func(inputs_shape, layer_class, modelname):
    inputs = ftensor5()
    mlp = construct_model(inputs_shape, layer_class, modelname)
    result = mlp.fprop(inputs)
    cost = T.sum(result)
    grad = T.grad(cost, mlp.layers[0].get_params()[0])
    
    grad = gpu_contiguous(grad)
    grad_func = theano.function([inputs], grad)
    return grad_func

def construct_model(inputs_shape, layer_class, modelname):
    filedir = os.path.join(os.path.dirname(__file__), 'mlps.yaml')
    layer_args = yaml_parse.load_path(filedir)[modelname]
    layers = []
    for i, layer_arg in enumerate(layer_args):
        layer = layer_class(irange=1e-3, layer_name='test_' + str(i),
            nonlinearity=IdentityConvNonlinearity(), **layer_arg)
        layers.append(layer)
    conv_3d_input_space = Conv3DSpace(inputs_shape[1:4], 
        num_channels=inputs_shape[4], axes=('b',0,1,2,'c'))
    mlp = MLP(input_space=conv_3d_input_space, layers=layers)
    return mlp

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="""Performance experiments for MLPs of 3d convolution layers.
        Example: perf/perf_single_3d_layer.py --inputs 15 3 4 5 1 
        --filters 12 3 4 5 1 --maxframes 30 --layer cudnn"""
    )
    parser.add_argument('--inputs', nargs='*', default=[3, 8, 8, 8, 3],
                        dest='inputs_shape',
                        help='''Shape of the inputs b 0 1 2 c format.
                        Time dimension will be determined by timemax parameter''')
    parser.add_argument('--layer', default='cudnn', dest='layername',
                        choices=_layername_to_class_dict.keys(),
                        help='''Layer to perf.''')
    parser.add_argument('--model', default='simple', dest='modelname',
                    choices=['simple', 'twolayer', 'twolayerpool'],
                    help='''Model to perf.''')
    args = parser.parse_args()
    args.inputs_shape = [int(s) for s in args.inputs_shape]
    args.layer_class = layername_to_class(args.layername)
    return args

_layername_to_class_dict = {
        'cudnn': CuDnnConv3dElemwise,
        'cublas': CuBlasConv3dElemwise,
    }

def layername_to_class(name):
    return _layername_to_class_dict[name]

if __name__ == "__main__":
    args = parse_command_line_arguments()
    perf_mlp(args.inputs_shape, args.layer_class, args.modelname)