#include <cudnn.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
void checkCudaErrors(cudaError_t status) {
    if (status != 0) {
      printf("Cuda failure: %s\n", cudaGetErrorString(status));
      assert(false);
    }
}

void checkCudnnErrors(cudnnStatus_t status) {
    if (status != 0) {
      printf("Cudnn failure: %s\n", cudnnGetErrorString(status));
      assert(false);
    }
}

size_t productOfDimensions(int dims[], int nbDims) {
  size_t product = 1;
  for (int i = 0; i < nbDims; ++i) {
    product *= dims[i];
  }
  return product;
}

void computeStrides(const int* dims, int nbDims, int* strides) {
  size_t strides_multiplied = 1;
  for (int i = nbDims - 1; i >= 0; --i) {
    strides[i] = strides_multiplied;
    strides_multiplied *= dims[i];
  }
}


int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage:\n./volumeric_in_c [device_number]\n");
    return 1;
  }
  // Set device
  cudaSetDevice(atoi(argv[1]));
  srand(489579847); // some random seed
  // Create variables
  cudnnHandle_t context_handle;
  cudnnTensorDescriptor_t input_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnCreate(&context_handle);
  cudnnCreateTensorDescriptor(&input_desc);
  cudnnCreateFilterDescriptor(&filter_desc);
  cudnnCreateTensorDescriptor(&output_desc);
  cudnnConvolutionDescriptor_t convDesc;
  int nbDims = 5;
  // Set variable dimensions
  // b c 0 1 2 format
  int inputDimA[] = {32, 3, 80, 80, 40};
  //int inputDimA[] = {64, 3, 80, 80, 40};
  int filterDimA[] = {32, 3, 5, 5, 5};
  //int filterDimA[] = {64, 3, 5, 5, 5};
  int filterStrideA[] = {1,1,1};
  int convDims = 3;
  int padA[] = {0,0,0};
  int upscaleA[] = {1,1,1};
  int inputStrideA[nbDims];
  computeStrides(inputDimA, nbDims, inputStrideA);
  cudnnCreateConvolutionDescriptor(&convDesc);
  checkCudnnErrors(cudnnSetConvolutionNdDescriptor(convDesc, convDims, padA,
      filterStrideA, upscaleA, CUDNN_CONVOLUTION));

  checkCudnnErrors(cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT,
      nbDims, inputDimA, inputStrideA));
  checkCudnnErrors(cudnnSetFilterNdDescriptor(filter_desc, CUDNN_DATA_FLOAT,
      nbDims, filterDimA));

  int outputDimA[nbDims];
  cudnnGetConvolutionNdForwardOutputDim(convDesc, input_desc, filter_desc,
      nbDims, outputDimA);
  int outputStrideA[nbDims];
  computeStrides(outputDimA, nbDims, outputStrideA);

  checkCudnnErrors(cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT,
      nbDims, outputDimA, outputStrideA));


  cudnnConvolutionFwdAlgo_t forward_algo;
  checkCudnnErrors(cudnnGetConvolutionForwardAlgorithm(context_handle,
      input_desc,
      filter_desc, convDesc, output_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
      0, &forward_algo));


  size_t worksize;
  checkCudnnErrors(cudnnGetConvolutionForwardWorkspaceSize(context_handle, input_desc,
      filter_desc, convDesc, output_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
      &worksize));
  assert(worksize == 0);

  // Allocata data memory
  float *srcData = NULL, *filterData = NULL, *dstData = NULL;
  size_t inputTotalDimension = productOfDimensions(inputDimA, nbDims);
  size_t filterTotalDimension = productOfDimensions(filterDimA, nbDims);
  size_t outputTotalDimension = productOfDimensions(outputDimA, nbDims);


  // Fill with values
  float* inputHost = new float[inputTotalDimension];
  for (int i = 0; i < inputTotalDimension; ++i) {
    inputHost[i] = (float)rand()/(float)(RAND_MAX);
  }
  float* filterHost = new float[filterTotalDimension];
  for (int i = 0; i < filterTotalDimension; ++i) {
    filterHost[i] = (float)rand()/(float)(RAND_MAX);
  }

  checkCudaErrors(cudaMalloc(&filterData, filterTotalDimension*sizeof(float)));
  checkCudaErrors(cudaMalloc(&dstData, outputTotalDimension*sizeof(float)));

  checkCudaErrors(cudaMalloc(&srcData, inputTotalDimension*sizeof(float)));

  cudaMemcpy(filterData, filterHost, filterTotalDimension*sizeof(float),
      cudaMemcpyHostToDevice);
  struct timeval tval_before, tval_after, tval_result;
  gettimeofday(&tval_before, NULL);
  cudaMemcpy(srcData, inputHost, inputTotalDimension*sizeof(float),
      cudaMemcpyHostToDevice);
  delete filterHost;
  delete inputHost;
  float alpha = 1;
  float beta = 0;

  checkCudnnErrors(cudnnConvolutionForward(context_handle, &alpha, input_desc,
      srcData, filter_desc, filterData, convDesc,
      CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, NULL, 0, &beta, output_desc,
      dstData));

  gettimeofday(&tval_after, NULL);
  timersub(&tval_after, &tval_before, &tval_result);
  double timeDiffInMs = (tval_result.tv_sec * 1000) +
      (tval_result.tv_usec / 1000.0);
  printf("Time elapsed %5.2f ms\n", timeDiffInMs);
  float* result = new float[outputTotalDimension];


  checkCudaErrors( cudaMemcpy(result, dstData,
      outputTotalDimension*sizeof(float), cudaMemcpyDeviceToHost));

  // prevent that computation is compiled away by optimization
  double checkSumForPerf = 0;
  for (int i = 0; i < outputTotalDimension; ++i) {
    checkSumForPerf += result[i];
  }
  printf("random checksum %f\n", checkSumForPerf);
  delete result;
}
// TODELAY: delete this?
// check for output dim
/*
int tensorOutputDimA[5];

err = cudnnGetConvolutionNdForwardOutputDim(convDesc, input_desc, filter_desc,
    nbDims, tensorOutputDimA);
printf("Output: ");
int i;
for (i = 0; i < 5; ++i) {
  printf("%d ", tensorOutputDimA[i]);
}
printf("\n");
printf("success getting output dim: %d\n", CUDNN_STATUS_SUCCESS == err);*/
