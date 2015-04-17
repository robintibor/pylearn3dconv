#include <cudnn.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <assert.h>
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


int main(int argc, char** argv) {
  // Set device
  cudaSetDevice(0);
  // Create variables
  cudnnHandle_t context_handle;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnPoolingDescriptor_t pool_desc;
  cudnnCreate(&context_handle);
  cudnnCreateTensorDescriptor(&input_desc);
  cudnnCreateTensorDescriptor(&output_desc);
  cudnnCreatePoolingDescriptor(&pool_desc);
  // Set variable dimensions
  // b c 0 1 2 format
  cudnnPoolingMode_t pool_mode = CUDNN_POOLING_MAX;
  float alpha = 1;
  float beta = 0;

  int nbDims = 5;
  int poolDims = 3;
  // changing to nbDims = 4 and poolDims = 2 makes it work.
  int inputDimA[5] = {1,1,1,1,1};
  int outputDimA[5] = {1,1,1,1,1};

  int poolShapeA[3] = {1,1,1};
  int poolPadA[3] = {0,0,0};
  int poolStrideA[3] = {1,1,1};


  int inputStrideA[5] = {1,1,1,1,1};
  int outputStrideA[5] = {1,1,1,1,1};


  checkCudnnErrors(cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT,
      nbDims, inputDimA, inputStrideA));

  checkCudnnErrors(cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT,
      nbDims, outputDimA, outputStrideA));


  // Allocata data memory and fill with values
  float *srcData = NULL, *filterData = NULL, *dstData = NULL;


  float* inputHost = new float[1];
  inputHost[0] = 1;
  float* outputHost = new float[1];
  outputHost[0] = 1;




  // Allocate memory on graphics card
  checkCudaErrors(cudaMalloc(&srcData, 1*sizeof(float)));
  checkCudaErrors(cudaMalloc(&dstData, 1*sizeof(float)));

// Copy values to graphics card
  checkCudaErrors(cudaMemcpy(srcData, inputHost, 1*sizeof(float),
      cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dstData, outputHost, 1*sizeof(float),
      cudaMemcpyHostToDevice));

  checkCudnnErrors(cudnnSetPoolingNdDescriptor(
    pool_desc,
    pool_mode,
    poolDims,
    poolShapeA,
    poolPadA,
    poolStrideA
    ));
  checkCudnnErrors(cudnnPoolingForward(
          context_handle,
          pool_desc,
          &alpha,
          input_desc, srcData,
          &beta,
          output_desc, dstData
    ));


  delete inputHost;
  delete outputHost;

  return 0;

}
