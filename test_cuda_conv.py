import os
import pycuda.driver as cuda
assert 'CUDA_DEVICE' in os.environ, "please set cuda device nr " + \
    "in environment for cluster"
import pycuda.autoinit
print("Using GPU Nr: {:d}".format(pycuda.autoinit.device.get_attribute(
    pycuda.driver.device_attribute.PCI_DEVICE_ID)))

from pycuda.compiler import SourceModule

from math import floor
from test_data import generate_test_data
import numpy as np
from convolutions import vectorized_conv

conv_mod = SourceModule(open('conv.cu').read())
loop_conv_on_gpu = conv_mod.get_function("loop_conv")

## also try with http://documen.tician.de/pycuda/tutorial.html -> Prepared Invocations
"""
grid = (1, 1)
block = (4, 4, 1)
func.prepare("P")
func.prepared_call(grid, block, a_gpu)"""

def loop_gpu_conv (inputs, filters, bias):
    num_batches = inputs.shape[0]
    out_height = inputs.shape[1] - filters.shape[1] + 1;
    out_width = inputs.shape[2] - filters.shape[2] + 1;
    out_duration = inputs.shape[3] - filters.shape[3] + 1;
    num_filters = filters.shape[0]
    output = np.zeros((num_batches, out_height, out_width, out_duration, 
        num_filters)).astype(np.float32)
    #print output
    loop_conv_on_gpu(cuda.In(inputs), cuda.In(filters),
        cuda.In(bias), cuda.InOut(output),
        np.int32(inputs.shape[0]),np.int32(inputs.shape[1]),
        np.int32(inputs.shape[2]),
        np.int32(inputs.shape[3]),np.int32(inputs.shape[4]),
        np.int32(filters.shape[0]),np.int32(filters.shape[1]),
        np.int32(filters.shape[2]), np.int32(filters.shape[3]),
        np.int32(filters.shape[4]),
        block=(1,1,1))
    #print output.shape
    #print output
    return output

def test_cuda_conv(rng):
    inputs, filters, bias = generate_test_data(rng)
    #bias *= 0. # for now to make it easier for debugging :)
    expected_result = vectorized_conv(inputs, filters, bias)
    loop_conv_gpu_result = loop_gpu_conv(inputs, filters, bias)
    print np.sum(np.square(loop_conv_gpu_result - expected_result))
    """
    diff = np.abs(loop_conv_gpu_result - expected_result)
    for batch_i in xrange(0, diff.shape[0]):
     for out_x in xrange(0, diff.shape[1]):
        for out_y in xrange(0, diff.shape[2]):
            for out_z in xrange(0, diff.shape[3]):
                for filt_i in xrange(0, diff.shape[4]):
                    this_diff = diff[batch_i][out_x][out_y][out_z][filt_i]
                    if (this_diff > 1e-6):
                        print diff[batch_i][out_x][out_y][out_z][filt_i]
                        print loop_conv_gpu_result[batch_i][out_x][out_y][out_z][filt_i]
                        print expected_result[batch_i][out_x][out_y][out_z][filt_i]
    print np.allclose(expected_result.astype('float32'), loop_conv_gpu_result)"""
if __name__ == "__main__":
    test_cuda_conv(np.random)# run your own 3d conv for comparison(!)
   
# call cuda function 

## Old THINGS::

mod = SourceModule(open('doublify.cu').read())
double_on_gpu = mod.get_function("doublify")
def double_the_array(a):
    b = np.empty_like(a).astype(np.float32)
    num_threads = 200
    num_elem_per_thread = np.int32(np.ceil(b.size / float(num_threads)))
    double_on_gpu(cuda.InOut(b), num_elem_per_thread, 
        np.int32(b.size), block=(num_threads, 1, 1))
    return b

check_dim_mod =  SourceModule("""
__global__ void check_dims(float *a) {
  int idx = threadIdx.x;
  a[idx] = idx;
}""")
check_dim = check_dim_mod.get_function('check_dims')

"""
 result = np.zeros((4,3,7)).astype(np.float32)
    check_dim(cuda.InOut(result),  block=(4*3*7,1,1))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            for k in range(result.shape[2]):
                print result[i][j][k]
    print result
"""
    