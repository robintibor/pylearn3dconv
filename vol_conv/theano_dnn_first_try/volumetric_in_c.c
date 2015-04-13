#include <cudnn.h>
#include <stdio.h>

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage:\n./volumeric_in_c [device_number]\n");
    return 1;
  }
  cudaSetDevice(atoi(argv[1]));
  cudnnHandle_t context_handle;
  cudnnStatus_t err = cudnnCreate(&context_handle);
  printf("success creating handle: %d\n", CUDNN_STATUS_SUCCESS == err);
  cudnnTensorDescriptor_t input_desc;
  err = cudnnCreateTensorDescriptor(&input_desc);
  printf("success creating input desc: %d\n", CUDNN_STATUS_SUCCESS == err);
  cudnnFilterDescriptor_t filter_desc;
  err = cudnnCreateFilterDescriptor(&filter_desc);
  printf("success creating filter desc: %d\n", CUDNN_STATUS_SUCCESS == err);
  cudnnTensorDescriptor_t output_desc;
  err = cudnnCreateTensorDescriptor(&output_desc);
  printf("success creating output desc: %d\n", CUDNN_STATUS_SUCCESS == err);
  int nbDims = 5;
  int dimA[] = {3, 3, 3, 3, 3};
  int filterDimA[] = {3, 3, 3, 3, 3};
  int strideA[] = {81, 27, 9, 3, 1};
  err = cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, nbDims, dimA,
      strideA);
  printf("success setting input desc: %d\n", CUDNN_STATUS_SUCCESS == err);
  err = cudnnSetFilterNdDescriptor(filter_desc, CUDNN_DATA_FLOAT, nbDims,
      filterDimA);
  // taken from python output
  int outputDimA[] = {3,3,1,1,1};
  int outputStrideA[] = {3,1,1,1,1};
  printf("success setting filter desc: %d\n", CUDNN_STATUS_SUCCESS == err);
  err = cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, nbDims,
      outputDimA, outputStrideA);
  printf("success setting output desc: %d\n", CUDNN_STATUS_SUCCESS == err);

  int convDims = 3;
  int padA[] = {0,0,0};
  int filterStrideA[] = {1,1,1};
  int upscaleA[] = {1,1,1};
  cudnnConvolutionDescriptor_t convDesc;
  err = cudnnCreateConvolutionDescriptor(&convDesc);
  printf("success creating conv desc: %d\n", CUDNN_STATUS_SUCCESS == err);
  err = cudnnSetConvolutionNdDescriptor(convDesc, convDims, padA,
      filterStrideA, upscaleA, CUDNN_CONVOLUTION);
  printf("success setting conv desc: %d\n", CUDNN_STATUS_SUCCESS == err);


  // check for output dim

  int tensorOutputDimA[5];

  err = cudnnGetConvolutionNdForwardOutputDim(convDesc, input_desc, filter_desc,
      nbDims, tensorOutputDimA);
  printf("Output: ");
  int i;
  for (i = 0; i < 5; ++i) {
    printf("%d ", tensorOutputDimA[i]);
  }
  printf("\n");
  printf("success getting output dim: %d\n", CUDNN_STATUS_SUCCESS == err);
  cudnnConvolutionFwdAlgo_t forward_algo;
  err = cudnnGetConvolutionForwardAlgorithm(context_handle, input_desc,
      filter_desc, convDesc, output_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
      0, &forward_algo);
  printf("success getting forward algorithm: %d\n", CUDNN_STATUS_SUCCESS == err);
  printf("forward algorithm %d \n", forward_algo);


  size_t worksize;
  err = cudnnGetConvolutionForwardWorkspaceSize(context_handle, input_desc,
      filter_desc, convDesc, output_desc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
      &worksize);
  printf("success getting workspace size: %d\n", CUDNN_STATUS_SUCCESS == err);
  printf("Workspace size: %zu\n", worksize);
  printf("%s\n", cudnnGetErrorString(err));
  printf("%d\n", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);
  printf("%d\n", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
  printf("%d\n", CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
  printf("%d\n", CUDNN_CONVOLUTION_FWD_ALGO_DIRECT);

//CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
}
