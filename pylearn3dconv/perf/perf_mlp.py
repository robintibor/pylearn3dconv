from pylearn3dconv.test import ftensor5
from numpy.random import RandomState
import numpy as np
from pylearn3dconv.volumetric_space import Conv3DSpace
from pylearn3dconv.layers.variants import CuDnnConv3dElemwise,\
    CuBlasConv3dElemwise, Theano3dConv3dElemwise, Theano3d2dConv3dElemwise,\
    TheanoFFTConv3dElemwise
from pylearn2.models.mlp import MLP, IdentityConvNonlinearity, ConvElemwise
import theano.tensor as T
from pylearn3dconv.perf import perf_func_print_results
import theano
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.config import yaml_parse
import os
import argparse
from pylearn2.space import Conv2DSpace

def perf_mlp(inputs_shape, layer_class, modelname, minframes, maxframes):
    rng = RandomState(np.uint32(hash('perfthemlp')))
    print "Perfing {:s}".format(layer_class.__name__)
    mlp_perfer = MLPPerf(inputs_shape, layer_class, modelname, minframes,
        maxframes, rng)
    mlp_perfer.perf()
    print (range(minframes, maxframes + 1))
    print(mlp_perfer.runtime_ms)

class MLPPerf():
    def __init__(self, inputs_shape, layer_class, modelname, minframes,
        maxframes, rng):
        self.__dict__.update(locals())
        del self.self

    def perf(self):
        self.runtime_ms = []
        for frames in range(self.minframes, self.maxframes + 1):
            print("Frames: {:d}".format(frames))
            self.inputs_shape[3] = frames
            self.grad_func = self.create_mlp_grad_func()
            inputs = self.generate_inputs()
            runtime_sec, iterations = perf_func_print_results(self.modelname,
                self.grad_func, None, inputs)
            self.runtime_ms.append(runtime_sec * 1000.0 / iterations)
            
    def generate_inputs(self):
        inputs_shape = self.inputs_shape
        if (self.layer_class == ConvElemwise):
            # eliminate time dimension
            inputs_shape = inputs_shape[0:3] + [inputs_shape[4]]
        inputs = self.rng.normal(size=inputs_shape).astype(np.float32)
        return inputs
    
    def create_mlp_grad_func(self):
        inputs = (ftensor5() if self.layer_class != ConvElemwise 
            else T.ftensor4())
        mlp = self.construct_model()
        result = mlp.fprop(inputs)
        cost = T.sum(result)
        grad = T.grad(cost, mlp.layers[0].get_params()[0])
        
        grad = gpu_contiguous(grad)
        grad_func = theano.function([inputs], grad)
        return grad_func
        
    def construct_model(self):
        filedir = os.path.join(os.path.dirname(__file__), 'mlps.yaml')
        layer_args = yaml_parse.load_path(filedir)[self.modelname]
        layers = []
        
        # adapt in case of 2d layer
        if (self.layer_class == ConvElemwise):
            self.adapt_for_2d_conv(layer_args)
        else:
            self.adapt_for_time_dim(layer_args)
        print layer_args
            
        for i, layer_arg in enumerate(layer_args):
            # maybe time dim smaller 
            layer = self.layer_class(irange=1e-3, layer_name='test_' + str(i),
                nonlinearity=IdentityConvNonlinearity(), **layer_arg)
            layers.append(layer)
        input_space = self.create_input_space()
        mlp = MLP(input_space=input_space, layers=layers)
        return mlp
        
    def create_input_space(self):
        if (self.layer_class != ConvElemwise):
            return Conv3DSpace(self.inputs_shape[1:4], 
                num_channels=self.inputs_shape[4], axes=('b',0,1,2,'c'))
        else:
            return Conv2DSpace(self.inputs_shape[1:3], 
                num_channels=self.inputs_shape[4], axes=('b',0,1,'c'))
        
        
    def adapt_for_2d_conv(self, layer_args):
        for layer_arg in layer_args:
            for arg in ['kernel_shape', 'kernel_stride', 'pool_shape', 
                        'pool_stride']:
                if arg in layer_arg:
                    layer_arg[arg] = layer_arg[arg][0:2]
    
    def adapt_for_time_dim(self, layer_args):
        # input shape is b012c
        # all shapes/strides here refer to the time dimension
        inshape = self.inputs_shape[3]
        for larg in layer_args:
            # Adjust kernel if necessary
            larg['kernel_shape'][2] = min(larg['kernel_shape'][2], inshape)
            kernel_shape = larg['kernel_shape'][2]
            kernel_stride = larg['kernel_stride'][2]
            inshape = ((inshape - kernel_shape) // kernel_stride) + 1
            
            # Adjust pooling if necessary
            if larg['pool_type'] is not None:
                larg['pool_shape'][2] = min(larg['pool_shape'][2], inshape)
                pool_shape = larg['pool_shape'][2]
                pool_stride = larg['pool_stride'][2]
                inshape = ((inshape - pool_shape) // pool_stride) + 1
        
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
                    choices=['simple', 'twolayer', 'twolayerpool'],
                    help='''Model to perf.''')
    parser.add_argument('--minframes', type=int, help='''Minimum dim for 
        second(time) dimension.''', default=6)
    parser.add_argument('--maxframes', type=int, help='''Maximum dim for 
        second(time) dimension.''', default=7)
    args = parser.parse_args()
    args.inputs_shape = [int(s) for s in args.inputs_shape]
    args.layer_class = layername_to_class(args.layername)
    return args

_layername_to_class_dict = {
        'cudnn': CuDnnConv3dElemwise,
        'cublas': CuBlasConv3dElemwise,
        'theano3d': Theano3dConv3dElemwise,
        'theano3d2d': Theano3d2dConv3dElemwise,
        'theanofft': TheanoFFTConv3dElemwise,
        '2d': ConvElemwise
    }

def layername_to_class(name):
    return _layername_to_class_dict[name]

if __name__ == "__main__":
    #--inputs 32 80 80 40 3
    args = parse_command_line_arguments()
    perf_mlp(args.inputs_shape, args.layer_class, args.modelname,
        args.minframes, args.maxframes)