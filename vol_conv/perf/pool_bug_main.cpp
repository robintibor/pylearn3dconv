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
    printf("Usage:\n./pool_bug_main [device_number]\n");
    return 1;
  }
  // Set device
  cudaSetDevice(atoi(argv[1]));
  srand(489579847); // some random seed
  // Create variables
  cudnnHandle_t context_handle;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnPoolingDescriptor_t pool_desc;
  cudnnCreate(&context_handle);
  cudnnCreateTensorDescriptor(&input_desc);
  cudnnCreateTensorDescriptor(&output_desc);
  cudnnCreatePoolingDescriptor(&pool_desc);
    /*
  Input descriptor
  Float: 1
  nbDims: 5
  Dimensions: 5 3 4 8 2
  Strides: 192 64 16 2 1

  Output descriptor
  Float: 1
  nbDims: 5
  Dimensions: 5 3 2 1 2
  Strides: 12 4 2 2 1

  Pooling descriptor
  nbDims: 3
  Shape: 2 2 1
  Padding: 0 0 0
  Strides: 2 7 1
  */
  // Set variable dimensions
  // b c 0 1 2 format
  cudnnPoolingMode_t pool_mode = CUDNN_POOLING_MAX;
  float alpha = 1;
  float beta = 0;
  /* original bug version:
  int nbDims = 5;
  int poolDims = 3;
  int inputDimA[] = {5,3,4,8,2};
  int outputDimA[] = {5,3,2,1,2};

  int poolShapeA [] = {2, 2, 1};
  int poolPadA[] = {0,0,0};
  int poolStrideA [] = {2, 7, 1};
   */
  /* Working 2d version

  int nbDims = 4;
  int poolDims = 2;
  int inputDimA[] = {1,1,1,1};
  int outputDimA[] = {1,1,1,1};

  int poolShapeA [] = {1,1};
  int poolPadA[] = {0,0};
  int poolStrideA [] = {1, 1};

   */

  int nbDims = 5;
  int poolDims = 3;
  int inputDimA[5] = {4,4,10,10,10};
  int outputDimA[5] = {4,4,5,5,5};

  int poolShapeA [3] = {2,2,2};
  int poolPadA[3] = {0,0,0};
  int poolStrideA [3] = {2,2,2};


  int inputStrideA[nbDims];
  int outputStrideA[nbDims];
  computeStrides(inputDimA, nbDims, inputStrideA);
  computeStrides(outputDimA, nbDims, outputStrideA);


  checkCudnnErrors(cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT,
      nbDims, inputDimA, inputStrideA));

  checkCudnnErrors(cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT,
      nbDims, outputDimA, outputStrideA));


  // Allocata data memory and fill with values
  float *srcData = NULL, *filterData = NULL, *dstData = NULL;
  size_t inputTotalDimension = productOfDimensions(inputDimA, nbDims);
  size_t outputTotalDimension = productOfDimensions(outputDimA, nbDims);


  float* inputHost = new float[inputTotalDimension];
  for (int i = 0; i < inputTotalDimension; ++i) {
    //inputHost[i] = (float)rand()/(float)(RAND_MAX);
    inputHost[i] = 1;
  }
  float* outputHost = new float[outputTotalDimension];
  for (int i = 0; i < outputTotalDimension; ++i) {
    //outputHost[i] = (float)rand()/(float)(RAND_MAX);
    outputHost[i] = 1;
  }




  // Allocate memory on graphics card
  checkCudaErrors(cudaMalloc(&srcData, inputTotalDimension*sizeof(float)));
  checkCudaErrors(cudaMalloc(&dstData, outputTotalDimension*sizeof(float)));

// Copy values to graphics card
  checkCudaErrors(cudaMemcpy(srcData, inputHost, inputTotalDimension*sizeof(float),
      cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(dstData, outputHost, outputTotalDimension*sizeof(float),
      cudaMemcpyHostToDevice));
  //delete inputHost;
  //delete outputHost;

  checkCudnnErrors(cudnnSetPoolingNdDescriptor(
    pool_desc,
    pool_mode,
    poolDims,
    poolShapeA,
    poolPadA,
    poolStrideA
    ));





  // PRINTING START
  {
    cudnnDataType_t dataType;
    nbDims = -1;
    int nbDimsRequested = 5;
    int dimA[nbDimsRequested];
    int strideA[nbDimsRequested];
   cudnnGetTensorNdDescriptor(input_desc,
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


   cudnnGetTensorNdDescriptor(output_desc,
          5,
          &dataType,
          &nbDims,
          dimA,
          strideA);
      printf("Output descriptor\n");
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


      cudnnPoolingMode_t mode;
      nbDimsRequested = 3;
      int padA[nbDimsRequested];
      cudnnGetPoolingNdDescriptor(pool_desc,
      nbDimsRequested,
      &mode,
      &nbDims,
      dimA,
      padA,
      strideA);
      printf("Pooling descriptor\n");
      printf("nbDims: %d\n", nbDims);
      printf("mode: %d\n", mode);
      printf("Shape: ");
      for (int i = 0; i < nbDims; ++i) {
        printf("%d ", dimA[i]);
      }
      printf("\n");
      printf("Padding: ");
      for (int i = 0; i < nbDims; ++i) {
        printf("%d ", padA[i]);
      }
      printf("\n");
      printf("Strides: ");
      for (int i = 0; i < nbDims; ++i) {
        printf("%d ", strideA[i]);
      }
      printf("\n");


      }
  // PRINTING END
  checkCudnnErrors(cudnnPoolingForward(
          context_handle,
          pool_desc,
          &alpha,
          input_desc, srcData,
          &beta,
          output_desc, dstData
    ));



  return 0;
  /*

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
*/

}
