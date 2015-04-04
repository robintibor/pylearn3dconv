#include <stdio.h>

void set_device(char** argv) {
  int device_nr = atoi(argv[1]);
  printf("Using GPU Nr: %d\n", device_nr);
  cudaSetDevice (device_nr);
}
