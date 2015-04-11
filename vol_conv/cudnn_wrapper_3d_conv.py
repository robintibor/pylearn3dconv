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
frames_in = 5
height_filter = 5
width_filter = 5
frames_filter = 4
pad_h = 4
pad_w = 4
vertical_stride = 1
horizontal_stride = 1
upscalex = 1
upscaley = 1
alpha = 1.0
beta = 1.0

inputs_shape = [n_input, filters_in, height_in, width_in, frames_in]
filters_shape = [filters_out, filters_in, height_filter, width_filter, 
    frames_filter]
nbDims = len(filters_shape)
conv_dims = 3 #?
padding_shape = [0] * conv_dims #
filter_stride_shape = [1] * conv_dims
upscale_shape = [1] * conv_dims
tensor_stride = [1] * nbDims
#test_desc = libcudnn.cudnnCreateTensorDescriptor()
#libcudnn.cudnnSetTensorNdDescriptor(test_desc, data_type,
#    n_input, filters_in, height_in, width_in)
# Input tensor
X = gpuarray.to_gpu(np.random.rand(*inputs_shape).astype(np.float32))

# Filter tensor
filters = gpuarray.to_gpu(np.random.rand(*filters_shape).astype(np.float32))

# Descriptor for input
X_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptor(X_desc, data_type,
    nbDims, inputs_shape, tensor_stride)
# Filter descriptor
filters_desc = libcudnn.cudnnCreateFilterDescriptor()
libcudnn.cudnnSetFilterNdDescriptor(filters_desc, data_type, nbDims,
    filters_shape)

# Convolution descriptor
conv_desc = libcudnn.cudnnCreateConvolutionDescriptor()
libcudnn.cudnnSetConvolutionNdDescriptor(conv_desc, conv_dims,
    padding_shape,
    filter_stride_shape, upscale_shape,
    convolution_mode)

# Get output dimensions (first two values are n_input and filters_out)
output_shape = libcudnn.cudnnGetConvolutionNdForwardOutputDim(
    conv_desc, X_desc, filters_desc, nbDims)
print output_shape
# Output tensor
Y = gpuarray.empty(output_shape, np.float32)
Y_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensorNdDescriptor(Y_desc, data_type, nbDims,
    output_shape, tensor_stride)

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
Y_arr = np.array(Y.get(), dtype='float32')

#print np.array(Y)
# Clean up
libcudnn.cudnnDestroyTensorDescriptor(X_desc)
libcudnn.cudnnDestroyTensorDescriptor(Y_desc)
libcudnn.cudnnDestroyFilterDescriptor(filters_desc)
libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
libcudnn.cudnnDestroy(cudnn_context)