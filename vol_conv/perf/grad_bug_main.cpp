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
  int inputDimA[] = {5,3,2,4,7};
  int filterDimA[] = {6, 3, 2, 3, 5};
  int filterStrideA[] = {1,1,1};
  int convDims = 3;
  int padA[] = {0,0,0};
  int upscaleA[] = {1,1,1};
  int inputStrideA[] = {168,56,28,7,1};
  // int inputStrideA[nbDims];
  // computeStrides(inputDimA, nbDims, inputStrideA);
  cudnnCreateConvolutionDescriptor(&convDesc);
  checkCudnnErrors(cudnnSetConvolutionNdDescriptor(convDesc, convDims, padA,
      filterStrideA, upscaleA, CUDNN_CROSS_CORRELATION));

  checkCudnnErrors(cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT,
      nbDims, inputDimA, inputStrideA));
  checkCudnnErrors(cudnnSetFilterNdDescriptor(filter_desc, CUDNN_DATA_FLOAT,
      nbDims, filterDimA));

  int outputDimA[] = {5,6,1,2,3};
  //int outputDimA[nbDims];
  //cudnnGetConvolutionNdForwardOutputDim(convDesc, input_desc, filter_desc,
  //    nbDims, outputDimA);
  int outputStrideA[nbDims];
  computeStrides(outputDimA, nbDims, outputStrideA);

  checkCudnnErrors(cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT,
      nbDims, outputDimA, outputStrideA));


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
  float* outputHost = new float[outputTotalDimension];
  for (int i = 0; i < outputTotalDimension; ++i) {
    outputHost[i] = (float)rand()/(float)(RAND_MAX);
  }

  checkCudaErrors(cudaMalloc(&filterData, filterTotalDimension*sizeof(float)));
  checkCudaErrors(cudaMalloc(&dstData, outputTotalDimension*sizeof(float)));

  checkCudaErrors(cudaMalloc(&srcData, inputTotalDimension*sizeof(float)));

  cudaMemcpy(filterData, filterHost, filterTotalDimension*sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(srcData, inputHost, inputTotalDimension*sizeof(float),
      cudaMemcpyHostToDevice);
  cudaMemcpy(dstData, outputHost, outputTotalDimension*sizeof(float),
      cudaMemcpyHostToDevice);
  delete filterHost;
  delete inputHost;
  delete outputHost;
  float alpha = 1;
  float beta = 0;


  // PRINTING START
  int nbDimsRequested = 5;
  cudnnDataType_t dataType;
  nbDims = -1;
  int dimA[nbDimsRequested];
  int strideA[nbDimsRequested];
  cudnnGetTensorNdDescriptor(output_desc,
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
  checkCudnnErrors(cudnnGetFilterNdDescriptor(filter_desc,
          5,
          &dataType,
          &nbDims,
          dimA));
      printf("Float: %d\n", dataType == CUDNN_DATA_FLOAT);
      printf("nbDims: %d\n", nbDims);
      printf("Dimensions: ");
      for (int i = 0; i < nbDims; ++i) {
        printf("%d ", dimA[i]);
      }
      printf("\n");
      printf("\n");
    printf("Conv descriptor\n");
    convDims = -1;
    //int padA[3];
    //int filterStrideA[3];
    //int upscaleA[3];
    cudnnConvolutionMode_t mode;
    checkCudnnErrors(cudnnGetConvolutionNdDescriptor(convDesc,
            3,
            &convDims,
            padA,
            filterStrideA,
            upscaleA,
            &mode));
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


        checkCudnnErrors(cudnnGetTensorNdDescriptor(input_desc,
            5,
            &dataType,
            &nbDims,
            dimA,
            strideA));
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
  // now lets do backward / gradient
  // lets just put it on input
  // lets use fake top gradient, just use output itself
  // reuse allocated input memory for gradient(!)
  checkCudnnErrors(cudnnConvolutionBackwardData(context_handle,
      &alpha, filter_desc, filterData, output_desc, dstData, convDesc, &beta,
      input_desc, srcData));
  float* resultGrad = new float[inputTotalDimension];
  checkCudaErrors(cudaMemcpy(resultGrad, srcData,
      inputTotalDimension*sizeof(float), cudaMemcpyDeviceToHost));
  double checkSumForPerf = 0;
   for (int i = 0; i < inputTotalDimension; ++i) {
     checkSumForPerf += resultGrad[i];
   }
   printf("random checksum grad %f\n", checkSumForPerf);


}
