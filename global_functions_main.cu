#include <stdlib.h>
#include <stdio.h>
#include "set_device.cuh"

__global__ void kernel(int *array)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  printf("block id thread id: %d %3d\n", blockIdx.x, threadIdx.x);
  array[index] = 7;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage:\n./global_functions_main [device_number]\n");
    return 1;
  }
  set_device(argv);
  
  int num_elements = 256;

  int num_bytes = num_elements * sizeof(int);

  // pointers to host & device arrays
  int *device_array = 0;
  int *host_array = 0;

  // malloc a host array
  host_array = (int*)malloc(num_bytes);

  // cudaMalloc a device array
  cudaMalloc((void**)&device_array, num_bytes);

  int block_size = 128;
  int grid_size = num_elements / block_size;

  kernel<<<grid_size,block_size>>>(device_array);

  // download and inspect the result on the host:
  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  // print out the result element by element
  for(int i=0; i < num_elements; ++i)
  {
    printf("%d ", host_array[i]);
  }
  printf("\n");
 
  // deallocate memory
  free(host_array);
  cudaFree(device_array);
}