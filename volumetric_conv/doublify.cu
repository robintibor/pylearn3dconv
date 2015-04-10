__global__ void doublify(float *a, int elem_per_thread, int size) {
  int idx = threadIdx.x * elem_per_thread;// + threadIdx.y*32;
  int num_iters = min(size - idx, elem_per_thread);
  for (int i= 0; i < num_iters; i++, idx++) {
      a[idx] *= 2;
  }
}