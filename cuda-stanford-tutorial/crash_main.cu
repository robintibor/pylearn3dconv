#include <stdio.h>
#include "set_device.cuh"
#include <stdlib.h>


inline void check_cuda_errors(const char *filename, const int line_number)
{
#ifdef DEBUG
  cudaThreadSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
    exit(-1);
  }
#endif
}

__global__ void foo(int *ptr)
{
  *ptr = 7;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage:\n./crash_main [device_number]\n");
    return 1;
  }
  foo<<<1,1>>>(0);
  check_cuda_errors(__FILE__, __LINE__);

  return 0;
}