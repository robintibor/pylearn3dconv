import pycuda.autoinit
from pycuda import gpuarray
import libcudnn, ctypes
import numpy as np

# Create a cuDNN context
cudnn_context = libcudnn.cudnnCreate()

# Set some options and tensor dimensions
tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']
convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
convolution_fwd_pref = libcudnn.cudnnConvolutionFwdPreference['CUDNN_CONVOLUTION_FWD_PREFER_FASTEST']

n_input = 100
filters_in = 10
filters_out = 8
height_in = 20
width_in = 20
height_filter = 5
width_filter = 5
pad_h = 4
pad_w = 4
vertical_stride = 1
horizontal_stride = 1
upscalex = 1
upscaley = 1
alpha = 1.0
beta = 1.0

test_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptor(test_desc, data_type,
    n_input, filters_in, height_in, width_in)
# Input tensor
X = gpuarray.to_gpu(np.random.rand(n_input, filters_in, height_in, width_in)
    .astype(np.float32))

# Filter tensor
filters = gpuarray.to_gpu(np.random.rand(filters_out,
    filters_in, height_filter, width_filter).astype(np.float32))

# Descriptor for input
X_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensor4dDescriptor(X_desc, tensor_format, data_type,
    n_input, filters_in, height_in, width_in)
# Filter descriptor
filters_desc = libcudnn.cudnnCreateFilterDescriptor()
libcudnn.cudnnSetFilter4dDescriptor(filters_desc, data_type, filters_out,
    filters_in, height_filter, width_filter)

# Convolution descriptor
conv_desc = libcudnn.cudnnCreateConvolutionDescriptor()
libcudnn.cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w,
    vertical_stride, horizontal_stride, upscalex, upscaley,
    convolution_mode)

# Get output dimensions (first two values are n_input and filters_out)
_, _, height_output, width_output = libcudnn.cudnnGetConvolution2dForwardOutputDim(
    conv_desc, X_desc, filters_desc)

# Output tensor
Y = gpuarray.empty((n_input, filters_out, height_output, width_output), np.float32)
Y_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensor4dDescriptor(Y_desc, tensor_format, data_type, n_input,
    filters_out, height_output, width_output)

# Get pointers to GPU memory
X_data = ctypes.c_void_p(int(X.gpudata))
filters_data = ctypes.c_void_p(int(filters.gpudata))
Y_data = ctypes.c_void_p(int(Y.gpudata))

# Perform convolution
algo = libcudnn.cudnnGetConvolutionForwardAlgorithm(cudnn_context, X_desc,
    filters_desc, conv_desc, Y_desc, convolution_fwd_pref, 0)
libcudnn.cudnnConvolutionForward(cudnn_context, alpha, X_desc, X_data,
    filters_desc, filters_data, conv_desc, algo, None, 0, beta,
    Y_desc, Y_data)
print type(X)
print type(Y)

# Clean up
libcudnn.cudnnDestroyTensorDescriptor(X_desc)
libcudnn.cudnnDestroyTensorDescriptor(Y_desc)
libcudnn.cudnnDestroyFilterDescriptor(filters_desc)
libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
libcudnn.cudnnDestroy(cudnn_context)