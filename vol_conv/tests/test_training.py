#!/usr/bin/env python
# I don't know how to properly import python packages, so I do it like this ;)
# http://stackoverflow.com/a/9806045/1469195
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentparentdir = os.path.dirname(os.path.dirname(currentdir))

os.sys.path.insert(0,parentparentdir) 

from vol_conv.volumetric_space import Conv3DSpace
from vol_conv.layers.theano_3d_conv import Theano3dConv3dElemwise
from vol_conv.layers.theano_3d_2d_conv import Theano3d2dConv3dElemwise
from vol_conv.layers.cublas_3d_conv import CuBlasConv3dElemwise
import numpy as np
from pylearn2.models.mlp import IdentityConvNonlinearity
import theano
import theano.tensor as T
from vol_conv.volumetric_dense_design_matrix import VolumetricDenseDesignMatrix
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.models.mlp import MLP, Softmax
from pylearn2.format.target_format import OneHotFormatter
from numpy.random import RandomState
from vol_conv.test_data import generate_test_data

def setup_training(inputs_shape, filters_shape, kernel_stride, conv_layer_class):
    rng = RandomState(hash('tobipuma') % 4294967295)
    inputs, _, _ = generate_test_data(rng, inputs_shape, 
        filters_shape)
    train_set, valid_set, test_set = generate_datasets(inputs)
    mlp = construct_model(inputs_shape, filters_shape, kernel_stride, 
        conv_layer_class)
    mlp_fprop = construct_predict_function(mlp)
    algorithm = create_algorithm(mlp, train_set)
    return mlp_fprop, train_set, valid_set, test_set, algorithm

def generate_datasets(inputs):
    targets = np.zeros(inputs.shape[0]).astype('int')
    targets[::2] = 1 # every second target is class 1 others class 0
    inputs[targets == 1] = inputs[targets == 1] + 1
    target_formatter = OneHotFormatter(2)
    targets_one_hot = target_formatter.format(targets)
    train_set = VolumetricDenseDesignMatrix(topo_view=inputs[0:50], 
        y=targets_one_hot[0:50], axes=('b', 0, 1, 2, 'c'))
    valid_set = VolumetricDenseDesignMatrix(topo_view=inputs[50:75], 
        y=targets_one_hot[50:75], axes=('b', 0, 1, 2, 'c'))
    test_set = VolumetricDenseDesignMatrix(topo_view=inputs[75:100], 
        y=targets_one_hot[75:100], axes=('b', 0, 1, 2, 'c'))
    return train_set, valid_set, test_set

def construct_model(inputs_shape, filters_shape, kernel_stride,
    conv_layer_class):
    conv_3d_input_space = Conv3DSpace(inputs_shape[1:4], 
        num_channels=inputs_shape[4], axes=('b',0,1,2,'c'))
    conv_3d_layer = conv_layer_class(output_channels=filters_shape[0], 
        kernel_shape=filters_shape[1:4],
        layer_name='conv3d_lin', nonlinearity=IdentityConvNonlinearity(),
        irange=0.001)
    softmax_layer = Softmax(max_col_norm=2, layer_name='y',
        n_classes=2, istdev=.05)
    mlp = MLP(input_space=conv_3d_input_space, layers=[conv_3d_layer,
        softmax_layer])
    return mlp

def construct_predict_function(mlp):
    ftensor5 = T.TensorType('float32', (False,)*5)
    inputs_mlp_theano = ftensor5()
    mlp_fprop_result = mlp.fprop(inputs_mlp_theano)
    mlp_fprop = theano.function([inputs_mlp_theano], mlp_fprop_result)
    return mlp_fprop

def create_algorithm(mlp, train_set):
    algorithm = SGD(batch_size=20, learning_rate=0.1)
    algorithm.setup(mlp, train_set)
    return algorithm

def run_training(mlp_fprop, train_set, valid_set, test_set, algorithm, 
    expected_results):
    results = {'train': [], 'valid': [], 'test': []} # for printing
    for _ in xrange(5):
        for set_and_name in ((train_set, 'train'), 
            (valid_set,'valid'), (test_set, 'test')):
            dataset = set_and_name[0]
            name = set_and_name[1]
            result = mlp_fprop(dataset.get_topological_view())
            labels = np.argmax(result, axis=1)
            y_labels = np.argmax(dataset.y, axis=1)
            accuracy = np.sum(np.equal(y_labels, labels)) / float(len(labels))
            results[name].append(accuracy)
        algorithm.train(train_set)
   
    """ for debug, enable this...
    for setname in results:
        print("Training mismatch,\n" + \
            "Expect {:s} for class {:s} to be:\n{:s},\nGot:\n{:s}").format(
                setname,
                algorithm.model.layers[0].__class__.__name__,
                expected_results[setname], 
                np.round(results[setname], decimals=2).tolist())"""
    for setname in results:
        assert np.allclose(results[setname], expected_results[setname]), \
            ("Training mismatch,\n" + \
            "Expect {:s} for class {:s} to be:\n{:s},\nGot:\n{:s}").format(
                setname,
                algorithm.model.layers[0].__class__.__name__,
                expected_results[setname], 
                np.round(results[setname], decimals=2).tolist())

def expect_results(inputs_shape, filters_shape, kernel_stride, conv_layer_class, 
    expected_results):
    mlp_fprop, train_set, valid_set, test_set, algorithm = setup_training(
        inputs_shape, filters_shape, kernel_stride, conv_layer_class)
    run_training(mlp_fprop, train_set, valid_set, test_set, algorithm,
        expected_results)
    print conv_layer_class.__name__ + " - Ok."

def test_training():
    inputs_shape = [100,7,6,5,3]
    filters_shape = [11,4,3,2,3]
    kernel_stride = [1, 1, 1]
    expect_results(inputs_shape, filters_shape, kernel_stride,
        Theano3dConv3dElemwise,
        {'train': [0.3, 0.98, 1.0, 1.0, 1.0],
         'valid': [0.4, 0.92, 1.0, 1.0, 1.0],
         'test': [0.32, 0.84, 0.88, 0.92, 0.92],
       })
    expect_results(inputs_shape, filters_shape, kernel_stride,
        Theano3d2dConv3dElemwise,
        {'train': [0.36, 0.98, 1.0, 1.0, 1.0],
         'valid': [0.36, 0.92, 1.0, 1.0, 1.0],
         'test': [0.36, 0.84, 0.88, 0.92, 0.92],
       })
    expect_results(inputs_shape, filters_shape, kernel_stride,
        CuBlasConv3dElemwise,
        {'train': [0.5, 0.98, 1.0, 1.0, 1.0],
         'valid': [0.4, 0.92, 1.0, 1.0, 1.0],
         'test': [0.24, 0.84, 0.88, 0.92, 0.92],
       })


if __name__ == '__main__':
    test_training()