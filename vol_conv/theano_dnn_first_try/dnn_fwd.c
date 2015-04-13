#section support_code_struct

int
APPLY_SPECIFIC(conv_fwd)(CudaNdarray *input, CudaNdarray *kerns,
                         CudaNdarray *om, cudnnConvolutionDescriptor_t desc,
                         float alpha, float beta, CudaNdarray **output) {
  printf("inside conv fwd\n");
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
  if (c_set_tensor5d(input, APPLY_SPECIFIC(input)) == -1)
    return 1;
  if (c_set_filter5d(kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;
  printf("after conv fwd setting tensors/filters\n");

#ifdef CONV_INPLACE
  Py_XDECREF(*output);
  *output = om;
  Py_INCREF(*output);
#else
  if (CudaNdarray_prep_output(output, 5, CudaNdarray_HOST_DIMS(om)) != 0)
    return 1;
  if (beta != 0.0 && CudaNdarray_CopyFromCudaNdarray(*output, om))
    return 1;
#endif

  if (c_set_tensor5d(*output, APPLY_SPECIFIC(output)) == -1)
    return 1;

  {
    size_t worksize;
    void *workspace;
    int nbDimsRequested = 5;
    printf("Handle zu %zu\n", _handle);
    printf("Handle d %d\n", _handle);
    printf("Conv algo %d\n", CONV_ALGO);
    cudnnDataType_t dataType;
    int nbDims = -1;

    int dimA[nbDimsRequested];
    int strideA[nbDimsRequested];
    err= cudnnGetTensorNdDescriptor(APPLY_SPECIFIC(input),
        5,
        &dataType,
        &nbDims,
        dimA,
        strideA);
    printf("Input descriptor\n");
    printf("Float: %d\n", dataType == CUDNN_DATA_FLOAT);
    printf("nbDims: %d\n", nbDims);
    printf("Dimensions: ");
    for (int i = 0; i < nbDims; ++i) {
      printf("%d ", dimA[i]);
    }
    printf("\n");
    printf("Strides: ");
    for (int i = 0; i < nbDims; ++i) {
      printf("%d ", strideA[i]);
    }
    printf("\n");
    printf("\n");
    printf("Filter descriptor\n");
    err= cudnnGetFilterNdDescriptor(APPLY_SPECIFIC(kerns),
            5,
            &dataType,
            &nbDims,
            dimA);
        printf("Float: %d\n", dataType == CUDNN_DATA_FLOAT);
        printf("nbDims: %d\n", nbDims);
        printf("Dimensions: ");
        for (int i = 0; i < nbDims; ++i) {
          printf("%d ", dimA[i]);
        }
        printf("\n");
        printf("\n");
      printf("Conv descriptor\n");
      int convDims;
      int padA[3];
      int filterStrideA[3];
      int upscaleA[3];
      cudnnConvolutionMode_t mode;
      err= cudnnGetConvolutionNdDescriptor(desc,
              3,
              &convDims,
              padA,
              filterStrideA,
              upscaleA,
              &mode);
          printf("Float: %d\n", dataType == CUDNN_DATA_FLOAT);
          printf("conv dims: %d\n", convDims);
          printf("Paddings: ");
          for (int i = 0; i < convDims; ++i) {
            printf("%d ", padA[i]);
          }
          printf("\n");
          printf("filterStrideA: ");
          for (int i = 0; i < convDims; ++i) {
            printf("%d ", filterStrideA[i]);
          }
          printf("\n");
          printf("upscaleA: ");
          for (int i = 0; i < convDims; ++i) {
            printf("%d ", upscaleA[i]);
          }
          printf("\n");
          printf("mode convolution: %d\n", mode == CUDNN_CONVOLUTION);
          printf("mode correlation: %d\n", mode == CUDNN_CROSS_CORRELATION);

          printf("\n");

          err= cudnnGetTensorNdDescriptor(APPLY_SPECIFIC(output),
              5,
              &dataType,
              &nbDims,
              dimA,
              strideA);
          printf("Output\n");
          printf("Float: %d\n", dataType == CUDNN_DATA_FLOAT);
          printf("nbDims: %d\n", nbDims);
          printf("Dimensions: ");
          for (int i = 0; i < nbDims; ++i) {
            printf("%d ", dimA[i]);
          }
          printf("\n");
          printf("Strides: ");
          for (int i = 0; i < nbDims; ++i) {
            printf("%d ", strideA[i]);
          }
          printf("\n");
    if (err != CUDNN_STATUS_SUCCESS) {
      printf("error already on getting tensor descriptor");
    }

    printf("after cudnnGetTensorNdDescriptor\n");
    printf("handle %d\n", handle);
    err = cudnnGetConvolutionForwardWorkspaceSize(_handle,
                                                  APPLY_SPECIFIC(input),
                                                  APPLY_SPECIFIC(kerns),
                                                  desc,
                                                  APPLY_SPECIFIC(output),
                                                  CONV_ALGO,
                                                  &worksize);

    printf("after cudnnGetConvolutionForwardWorkspaceSize\n");
    printf("worksize requested: %zu\n", worksize);
    if (err != CUDNN_STATUS_SUCCESS) {
      PyErr_Format(PyExc_RuntimeError,
                   "GpuDnn3dConv: error getting worksize: %s",
                   cudnnGetErrorString(err));
      return 1;
    }

    printf("after cudnnGetConvolutionForwardWorkspaceSize error check\n");
    workspace = get_work_mem(worksize);
    if (workspace == NULL && worksize != 0)
      return 1;

    printf("before cudnnConvolutionForward\n");
    err = cudnnConvolutionForward(
      _handle,
      (void *)&alpha,
      APPLY_SPECIFIC(input), CudaNdarray_DEV_DATA(input),
      APPLY_SPECIFIC(kerns), CudaNdarray_DEV_DATA(kerns),
      desc,
      CONV_ALGO,
      workspace, worksize,
      (void *)&beta,
      APPLY_SPECIFIC(output), CudaNdarray_DEV_DATA(*output));
    printf("after cudnnConvolutionForward\n");
  }
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "GpuDnn3dConv: error doing operation: %s",
		 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
