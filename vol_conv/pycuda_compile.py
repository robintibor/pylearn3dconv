import pycuda.driver as cuda
import os
assert 'CUDA_DEVICE' in os.environ, "please set cuda device nr " + \
    "in environment for cluster"
import pycuda.autoinit
print("Using GPU Nr: {:d}".format(pycuda.autoinit.device.get_attribute(
    pycuda.driver.device_attribute.PCI_DEVICE_ID)))
from pycuda.compiler import SourceModule
from numpy.random import RandomState

import numpy

if __name__ == '__main__':
    rng = RandomState(298374)
    a = rng.randn(4,4)
    a = a.astype(numpy.float32)
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)
    mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
  """)
    func = mod.get_function("doublify")
    func(a_gpu, block=(4,4,1))
    a_doubled = numpy.empty_like(a)
    cuda.memcpy_dtoh(a_doubled, a_gpu)
    print a_doubled
    
    func(cuda.InOut(a), block=(4,4,1))
    print a
    mod_from_tutorial = SourceModule("""__global__ void kernel(int *array)
{
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  // map the two 2D indices to a single linear, 1D index
  int grid_width = gridDim.x * blockDim.x;
  int index = index_y * grid_width + index_x;

  // map the two 2D block indices to a single linear, 1D block index
  int result = blockIdx.y * gridDim.x + blockIdx.x;

  // write out the result
  array[index] = result;
}""")
    
    int_arr = numpy.zeros((16,16),dtype=numpy.int32)
    
    func_from_tutorial = mod_from_tutorial.get_function("kernel")
    
    func_from_tutorial(cuda.InOut(int_arr), grid=(4,4,1), block=(4,4,1))
    print int_arr