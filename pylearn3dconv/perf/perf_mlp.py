from pylearn3dconv.test import ftensor5
from numpy.random import RandomState
import numpy as np
from pylearn3dconv.volumetric_space import Conv3DSpace
from pylearn3dconv.layers.variants import CuDnnConv3dElemwise,\
    CuBlasConv3dElemwise, Theano3dConv3dElemwise, Theano3d2dConv3dElemwise
from pylearn2.models.mlp import MLP, IdentityConvNonlinearity, ConvElemwise
import theano.tensor as T
from pylearn3dconv.perf import perf_func_print_results
import theano
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.config import yaml_parse
import os
import argparse
from pylearn2.space import Conv2DSpace

def perf_mlp(inputs_shape, layer_class, modelname):
    rng = RandomState(np.uint32(hash('perfthemlp')))
    # generate mlp grad
    # generate inputs
    # perf....
    """
    mlp_perfer = MLPPerf(inputs_shape, layer_class, modelname)
    mlp_perfer.setup()
    mlp_perfer.perf()"""
    mlp_grad_func = create_mlp_grad_func(inputs_shape, layer_class, modelname)
    inputs = generate_inputs(rng, inputs_shape, layer_class)
    mlp_grad_func(inputs)
    perf_func_print_results(modelname, mlp_grad_func, None, inputs)
"""
class MLPPerf():
    def __init__(self, inputs_shape, layer_class, modelname):
        self.__dict__.update(locals())
        del self.self
        
    def setup(self):
        
    def perf(self):
"""        
        

def generate_inputs(rng, inputs_shape, layer_class):
    inputs = rng.normal(size=inputs_shape).astype(np.float32)
    if (layer_class == ConvElemwise):
        inputs = inputs[:,:,:,0,:] # eliminate time/3rd dimension
    return inputs

def create_mlp_grad_func(inputs_shape, layer_class, modelname):
    inputs = ftensor5() if layer_class != ConvElemwise else T.ftensor4()
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
        
        # adapt in case of 2d layer
        if (layer_class == ConvElemwise):
            adapt_for_2d_conv(layer_arg)
            
        layer = layer_class(irange=1e-3, layer_name='test_' + str(i),
            nonlinearity=IdentityConvNonlinearity(), **layer_arg)
        layers.append(layer)
    input_space = create_input_space(inputs_shape, layer_class)
    
    mlp = MLP(input_space=input_space, layers=layers)
    return mlp

def adapt_for_2d_conv(layer_arg):
    for arg in ['kernel_shape', 'kernel_stride', 'pool_shape', 
                'pool_stride']:
        if arg in layer_arg:
            layer_arg[arg] = layer_arg[arg][0:2]

def create_input_space(inputs_shape, layer_class):
    if (layer_class != ConvElemwise):
        return Conv3DSpace(inputs_shape[1:4], num_channels=inputs_shape[4],
            axes=('b',0,1,2,'c'))
    else:
        return Conv2DSpace(inputs_shape[1:3], num_channels=inputs_shape[4],
            axes=('b',0,1,'c'))

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(
        description="""Performance experiments for MLPs of 3d convolution layers.
        Example: perf/perf.mlp.py --inputs 15 3 4 5 1 --layer cudnn --model simple"""
    )
    parser.add_argument('--inputs', nargs='*', default=[3, 8, 8, 8, 3],
                        dest='inputs_shape',
                        help='''Shape of the inputs b 0 1 2 c format.
                        Time dimension will be determined by timemax parameter''')
    parser.add_argument('--layer', default='cudnn', dest='layername',
                        choices=_layername_to_class_dict.keys(),
                        help='''Layer to perf.''')
    parser.add_argument('--model', default='simple', dest='modelname',
                    choices=['simple', 'twolayer', 'twolayerpool', 'fake2d',
                        'fake2dpool'],
                    help='''Model to perf.''')
    args = parser.parse_args()
    args.inputs_shape = [int(s) for s in args.inputs_shape]
    args.layer_class = layername_to_class(args.layername)
    return args

_layername_to_class_dict = {
        'cudnn': CuDnnConv3dElemwise,
        'cublas': CuBlasConv3dElemwise,
        'theano3d': Theano3dConv3dElemwise,
        'theano3d2d': Theano3d2dConv3dElemwise,
        '2d': ConvElemwise
    }

def layername_to_class(name):
    return _layername_to_class_dict[name]

if __name__ == "__main__":
    #--inputs 32 80 80 40 3
    args = parse_command_line_arguments()
    perf_mlp(args.inputs_shape, args.layer_class, args.modelname)