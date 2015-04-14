from timeit import default_timer as timer
import numpy as np
def perf_func(name, func, correct_result, *func_args):
    total_running_time = 0
    runs = 0
    while (total_running_time < 1 or 
        (total_running_time < 2 and runs < 30)):
        start = timer()
        if (len(func_args) > 0):
            result = func(*func_args)
        else:
            result = func()
        end = timer()
        total_running_time += end - start
        runs += 1
    print("{:25s} runtime per iteration ({:4d} iterations): {:8.4f}ms".format(
        name, runs, 1000 * total_running_time / runs))
    if (correct_result is not None):
        diff = np.sum(np.square(correct_result - result))
        tolerance = 1e-2
        assert diff < tolerance, "Diff {:f} too big for {:s}".format(diff, name)
        if diff < tolerance:
            print("Result check ok with diff {:4.3f}".format(diff))
