from pylearn3dconv.layers.base import Conv3dElemwise
from pylearn3dconv.layers.conv_transformers import (CuDnn3dConv, CuBlas3dConv,
    Theano3dConv, Theano3d2dConv)
from pylearn3dconv.layers.pool_transformers import CudnnPoolTransformer

class CuDnnConv3dElemwise(Conv3dElemwise):
    conv_transformer=CuDnn3dConv
    pool_transformer=CudnnPoolTransformer

class CuBlasConv3dElemwise(Conv3dElemwise):
    conv_transformer=CuBlas3dConv
    pool_transformer=CudnnPoolTransformer

class Theano3dConv3dElemwise(Conv3dElemwise):
    conv_transformer=Theano3dConv
    pool_transformer=CudnnPoolTransformer

class Theano3d2dConv3dElemwise(Conv3dElemwise):
    conv_transformer=Theano3d2dConv
    pool_transformer=CudnnPoolTransformer