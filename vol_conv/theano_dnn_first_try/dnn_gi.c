#section support_code_struct

int
APPLY_SPECIFIC(conv_gi)(CudaNdarray *kerns, CudaNdarray *output,
                        CudaNdarray *im, cudnnConvolutionDescriptor_t desc,
                        float alpha, float beta, CudaNdarray **input) {
  cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
  if (c_set_tensor5d(output, APPLY_SPECIFIC(output)) == -1)
    return 1;
  // PRINTING SMALL OTHER THING
  int nbDims2 = 5;
  int dimA2[nbDims2];
  int strideA2[nbDims2];
  cudnnDataType_t dataType2;
  err= cudnnGetTensorNdDescriptor(APPLY_SPECIFIC(output),
      5,
      &dataType2,
      &nbDims2,
      dimA2,
      strideA2);
  printf("Output descriptor before (2nd arg)\n");
   printf("Float: %d\n", dataType2 == CUDNN_DATA_FLOAT);
   printf("nbDims: %d\n", nbDims2);
   printf("Dimensions: ");
   for (int i = 0; i < nbDims2; ++i) {
     printf("%d ", dimA2[i]);
   }
   printf("\n");
   printf("Strides: ");
   for (int i = 0; i < nbDims2; ++i) {
     printf("%d ", strideA2[i]);
   }
   printf("\n");
   // END PRINTING SMALL OTHER THING
  if (c_set_filter5d(kerns, APPLY_SPECIFIC(kerns)) == -1)
    return 1;

#ifdef CONV_INPLACE
  printf("dnn_gi out inplace\n");
  Py_XDECREF(*input);
  *input = im;
  Py_INCREF(*input);
#else
  printf("dnn_gi out of place\n");
  if (CudaNdarray_prep_output(input, 5, CudaNdarray_HOST_DIMS(im)) != 0)
    return 1;
  if (beta != 0.0 && CudaNdarray_CopyFromCudaNdarray(*input, im))
    return 1;
#endif

  if (c_set_tensor5d(*input, APPLY_SPECIFIC(input)) == -1)
    return 1;

  // PRINTING START
  int nbDimsRequested = 5;
  cudnnDataType_t dataType;
  int nbDims = -1;
  int dimA[nbDimsRequested];
  int strideA[nbDimsRequested];
  err= cudnnGetTensorNdDescriptor(APPLY_SPECIFIC(output),
      5,
      &dataType,
      &nbDims,
      dimA,
      strideA);
  printf("Output descriptor (2nd arg)\n");
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
  printf("Filter descriptor (1st arg)\n");
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


        err= cudnnGetTensorNdDescriptor(APPLY_SPECIFIC(input),
            5,
            &dataType,
            &nbDims,
            dimA,
            strideA);
        printf("Input (6th arg)\n");
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

  // PRINTING END

  err = cudnnConvolutionBackwardData(
    _handle,
    (void *)&alpha,
    APPLY_SPECIFIC(kerns), CudaNdarray_DEV_DATA(kerns),
    APPLY_SPECIFIC(output), CudaNdarray_DEV_DATA(output),
    desc,
    (void *)&beta,
    APPLY_SPECIFIC(input), CudaNdarray_DEV_DATA(*input));
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "GpuDnn3dConvGradI: error doing operation: %s",
                 cudnnGetErrorString(err));
    return 1;
  }
  return 0;
}
