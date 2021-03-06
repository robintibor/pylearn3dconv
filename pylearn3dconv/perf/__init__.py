from timeit import default_timer as timer
import numpy as np

import gc
def perf_func_print_results(name, func, correct_result, *func_args):
    runtime_in_sec, iterations = perf_func(func, correct_result, *func_args)
    print_results(name, runtime_in_sec, iterations)
    return runtime_in_sec, iterations

def perf_func(func, correct_result, *func_args):
    gc.collect()#make sure memory is empty
    ## 1 second warmup before actual computation
    warm_up_time = 0
    warm_up_runs = 0
    while ((warm_up_time < 1 and warm_up_runs < 50) or
            warm_up_runs < 2):
        start = timer()
        #print("memory before warmup call: {:5.1f} MB".format( 
        #    theano.sandbox.cuda.mem_info()[0] / (1024.0 ** 2)))
        if (len(func_args) > 0):
            result = func(*func_args)
        else:
            result = func()
        end = timer()
        assert result is not None # just make sure no optimizations remove computation

        del result
        gc.collect()#make sure memory is empty, here just to prevent gpu crashes
        warm_up_time += end - start
        warm_up_runs += 1
    total_running_time = 0
    runs = 0
    result = None
    while ((total_running_time < 1 and runs < 100) or 
        (total_running_time < 2 and runs < 30) or
        (total_running_time < 5 and runs < 20)
        or (total_running_time < 60 and runs < 5)):
        del result
        gc.collect()#make sure memory is empty
        #print("memory before real call:   {:5.1f} MB".format( 
        #    theano.sandbox.cuda.mem_info()[0] / (1024.0 ** 2)))
        start = timer()
        if (len(func_args) > 0):
            result = func(*func_args)
        else:
            result = func()
        end = timer()
        total_running_time += end - start
        runs += 1
    if (correct_result is not None):
        result = np.array(result)
        diff = np.sum(np.square(correct_result - result))
        tolerance = 1e-2 # has to be that high for theano2d3d apparently
        assert diff < tolerance, "Diff {:f} too big".format(diff)
        if diff < tolerance:
            print("Result check ok with diff {:4.3f}".format(diff))
    return total_running_time, runs 
    

def print_results(name, runtime_in_sec, iterations):
    print("{:25s} runtime per iteration ({:4d} iterations): {:8.4f}ms".format(
        name,  iterations, 1000 * runtime_in_sec / iterations))
