#section support_code
static cudnnHandle_t _handle = NULL;

static int
c_set_tensor5d(CudaNdarray *var, cudnnTensorDescriptor_t desc) {
    int nbDims = var->nd;
    if (nbDims != 5) {
      PyErr_Format(PyExc_RuntimeError,
                   "Number of dimensions should be 5 for 3d convolution, "
                   "instead got %d",
                   nbDims);
      return -1;
    }
    int hostStrides[nbDims];
    // multiply dims of remaining dims in case
    // there is one stride of dimension 0
    int dims_multiplied = 1;
    for (int i = nbDims - 1; i >= 0; i--) {
      hostStrides[i] = CudaNdarray_HOST_STRIDES(var)[i];
      if (hostStrides[i] == 0) {
        hostStrides[i] = dims_multiplied;
      }
      dims_multiplied *= CudaNdarray_HOST_DIMS(var)[i];
    }

    cudnnStatus_t err = cudnnSetTensorNdDescriptor(
    desc, CUDNN_DATA_FLOAT, nbDims, CudaNdarray_HOST_DIMS(var), hostStrides);

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
		 "Could not set tensor5d descriptor: %s"
		 "shapes=%d %d %d %d %d, strides=%d %d %d %d %d",
		 cudnnGetErrorString(err),
		 CudaNdarray_HOST_DIMS(var)[0],
		 CudaNdarray_HOST_DIMS(var)[1],
		 CudaNdarray_HOST_DIMS(var)[2],
		 CudaNdarray_HOST_DIMS(var)[3],
                 CudaNdarray_HOST_DIMS(var)[4],
                 hostStrides[0],
                 hostStrides[1],
                 hostStrides[2],
                 hostStrides[3],
                 hostStrides[4]
    );
  }
  return 0;
}

static int
c_set_filter5d(CudaNdarray *var, cudnnFilterDescriptor_t desc) {
  if (!CudaNdarray_is_c_contiguous(var)) {
    PyErr_SetString(PyExc_ValueError,
		    "Only contiguous filters (kernels) are supported.");
    return -1;
  }
  int nbDims = var->nd;
  if (nbDims != 5) {
    PyErr_Format(PyExc_RuntimeError,
                 "Number of dimensions should be 5 for 3d convolution, "
                 "instead got %d",
                 nbDims);
    return -1;
  }

  cudnnStatus_t err = cudnnSetFilterNdDescriptor(
    desc, CUDNN_DATA_FLOAT, nbDims, CudaNdarray_HOST_DIMS(var));

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
		 "Could not set filter descriptor: %s."
		 " dims= %d %d %d %d %d",
		 cudnnGetErrorString(err),
		 CudaNdarray_HOST_DIMS(var)[0],
		 CudaNdarray_HOST_DIMS(var)[1],
		 CudaNdarray_HOST_DIMS(var)[2],
		 CudaNdarray_HOST_DIMS(var)[3],
                 CudaNdarray_HOST_DIMS(var)[4]);
    return -1;
  }
  return 0;
}

#section init_code

{
  cudnnStatus_t err;
  if ((err = cudnnCreate(&_handle)) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not create cuDNN handle: %s",
		 cudnnGetErrorString(err));
#if PY_MAJOR_VERSION >= 3
    return NULL;
#else
    return;
#endif
  }
}
