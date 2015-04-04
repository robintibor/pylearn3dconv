#include "util/cuPrintf.cu"
#include <stdio.h>
#include "set_device.cuh"

__global__ void device_greetings(void)
{
  cuPrintf("Hello, world from the device!\n");
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage:\n./hello_world_main [device_number]\n");
    return 1;
  }
  set_device(argv);
  // greet from the host
  printf("Hello, world from the host!\n");

  // initialize cuPrintf
  cudaPrintfInit();

  // launch a kernel with a single thread to greet from the device
  // changed to two thread groups with 3 threads each => 6 greetings
  device_greetings<<<2,3>>>();

  // display the device's greeting
  cudaPrintfDisplay();
  
  // clean up after cuPrintf
  cudaPrintfEnd();

  return 0;
}