#!/usr/bin/env python
# I don't know how to properly import python packages, so I do it like this ;)
# http://stackoverflow.com/a/9806045/1469195
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentparentdir = os.path.dirname(os.path.dirname(currentdir))

os.sys.path.insert(0,parentparentdir) 

from volumetric_conv.volumetric_space import Conv3DSpace
from volumetric_conv.layers.theano_3d_conv import Theano3dConv3dElemwise
import numpy as np
from pylearn2.models.mlp import IdentityConvNonlinearity
import theano
import theano.tensor as T
from volumetric_conv.volumetric_dense_design_matrix import VolumetricDenseDesignMatrix
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.models.mlp import MLP, Softmax
from pylearn2.format.target_format import OneHotFormatter
from numpy.random import RandomState

def test_training():
    print("hihi")


if __name__ == '__main__':
    test_training()