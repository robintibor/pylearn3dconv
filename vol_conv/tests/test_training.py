#!/usr/bin/env python
# I don't know how to properly import python packages, so I do it like this ;)
# http://stackoverflow.com/a/9806045/1469195
import os
import inspect
from vol_conv.layers.cudnn_3d_conv import CuDnnConv3dElemwise
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

def test_training():
    inputs_shape = [100,7,6,5,3]
    filters_shape = [11,4,3,2,3]
    # First with no stride and no pooling
    print("No stride no pooling")
    kernel_stride = [1, 1, 1]
    pool_type = None
    pool_shape = [1,1,1]
    pool_stride = [1,1,1]
    expected_results = {'train': [0.58, 0.88, 0.96, 0.98, 1.0],
         'valid': [0.72, 0.76, 0.92, 0.96, 1.0],
         'test': [0.56, 0.6, 0.8, 0.96, 0.96],
       }
    
    expect_results(inputs_shape, filters_shape, kernel_stride,
        Theano3dConv3dElemwise, expected_results)
    expect_results(inputs_shape, filters_shape, kernel_stride,
        CuBlasConv3dElemwise, expected_results)
    expect_results(inputs_shape, filters_shape, kernel_stride,
        CuDnnConv3dElemwise, expected_results)

    # Then with stride
    print("Stride and no pooling")
    kernel_stride = [2, 1, 2]
    expected_results = {'train': [0.52, 0.92, 0.98, 1.0, 1.0],
         'valid': [0.6, 0.8, 0.84, 0.8, 0.84],
         'test': [0.48, 0.68, 0.84, 0.84, 0.84],
       }
    expect_results(inputs_shape, filters_shape, kernel_stride,
        Theano3dConv3dElemwise, expected_results)
    expect_results(inputs_shape, filters_shape, kernel_stride,
        CuBlasConv3dElemwise, expected_results)
    expect_results(inputs_shape, filters_shape, kernel_stride,
        CuDnnConv3dElemwise, expected_results)
    # Then with stride and fake pooling expect same results as without pooling
    print("Stride and fake pooling")
    pool_type = 'max'
    pool_shape = [1,1,1]
    pool_stride = [1,1,1]
    kernel_stride = [2, 1, 2]
    expected_results = {'train': [0.52, 0.92, 0.98, 1.0, 1.0],
         'valid': [0.6, 0.8, 0.84, 0.8, 0.84],
         'test': [0.48, 0.68, 0.84, 0.84, 0.84],
       }
    expect_results(inputs_shape, filters_shape, kernel_stride,
        Theano3dConv3dElemwise, expected_results)
    expect_results(inputs_shape, filters_shape, kernel_stride,
        CuBlasConv3dElemwise, expected_results)
    expect_results(inputs_shape, filters_shape, kernel_stride,
        CuDnnConv3dElemwise, expected_results)
    # Then with stride and real pooling
    print("real pooling...should fail as its not implemented")
    pool_type = 'max'
    pool_shape = [1,1,1]
    pool_stride = [1,1,1]
    kernel_stride = [2, 1, 2]
    expect_results(inputs_shape, filters_shape, kernel_stride,
        Theano3dConv3dElemwise,
        {'train': [0.62, 1.0, 1.0, 1.0, 1.0],
         'valid': [0.56, 0.92, 1.0, 1.0, 1.0],
         'test': [0.56, 0.88, 0.96, 1.0, 1.0],
       })
    expect_results(inputs_shape, filters_shape, kernel_stride,
        Theano3d2dConv3dElemwise,
        {'train': [0.6, 1.0, 1.0, 1.0, 1.0],
         'valid': [0.68, 0.92, 0.96, 1.0, 1.0],
         'test': [0.56, 0.84, 0.92, 1.0, 1.0],
       })
    expect_results(inputs_shape, filters_shape, kernel_stride,
        CuBlasConv3dElemwise,
        {'train': [0.66, 1.0, 1.0, 1.0, 1.0],
         'valid': [0.68, 0.88, 0.96, 1.0, 1.0],
         'test': [0.84, 0.84, 0.92, 0.96, 0.96],
       })


def expect_results(inputs_shape, filters_shape, kernel_stride, conv_layer_class, 
    expected_results):
    # a great seed is half the work :)
    rng = RandomState(hash('tobipuma') % 4294967295)
    inputs, filters, bias = generate_test_data(rng, inputs_shape, filters_shape)
    mlp_fprop, train_set, valid_set, test_set, algorithm = setup_training(
        inputs, filters, bias, kernel_stride, conv_layer_class)
    run_training(mlp_fprop, train_set, valid_set, test_set, algorithm,
        expected_results)
    print conv_layer_class.__name__ + " - Ok."

def setup_training(inputs, filters, bias, kernel_stride, conv_layer_class):
    """ Setup model, prediction function, algorithm for training"""
    train_set, valid_set, test_set = generate_datasets(inputs)
    mlp = construct_model(inputs.shape, filters, bias, kernel_stride, 
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

def construct_model(inputs_shape, filters, bias, kernel_stride,
    conv_layer_class):
    conv_3d_input_space = Conv3DSpace(inputs_shape[1:4], 
        num_channels=inputs_shape[4], axes=('b',0,1,2,'c'))
    conv_3d_layer = conv_layer_class(output_channels=filters.shape[0], 
        kernel_shape=filters.shape[1:4], kernel_stride = kernel_stride,
        layer_name='conv3d_lin', nonlinearity=IdentityConvNonlinearity(),
        irange=0.001)
    softmax_layer = Softmax(max_col_norm=2, layer_name='y',
        n_classes=2, istdev=.05)
    mlp = MLP(input_space=conv_3d_input_space, layers=[conv_3d_layer,
        softmax_layer])
    # convert filters to correct axes (('b', 0, 1, 2, ' c') are test data axes)
    converted_filters = Conv3DSpace.convert_numpy(filters, 
        ('b', 0, 1, 2, 'c'), conv_3d_layer.detector_space.axes)
    conv_3d_layer.set_weights(converted_filters)
    conv_3d_layer.set_biases(bias)
    return mlp

def construct_predict_function(mlp):
    ftensor5 = T.TensorType('float32', (False,)*5)
    inputs_mlp_theano = ftensor5()
    mlp_fprop_result = mlp.fprop(inputs_mlp_theano)
    mlp_fprop = theano.function([inputs_mlp_theano], mlp_fprop_result)
    return mlp_fprop

def create_algorithm(mlp, train_set):
    rng = RandomState(hash('tobipuma') % 4294967295)
    algorithm = SGD(batch_size=20, learning_rate=0.1)
    algorithm.rng = rng #try to always have same results for algorithm
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
   
    """for debug, enable this...
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

if __name__ == '__main__':
    test_training()