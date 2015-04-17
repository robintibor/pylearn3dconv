import numpy as np
def test_function(function, name, reference_result, *func_args):
    if (len(func_args) > 0):
            result = function(*func_args)
    else:
        result = function()
    if (not np.allclose(result, reference_result,
        atol=1e-3, rtol=0)):
        diff = np.sum(np.abs(result - reference_result))
        sum_of_squared_diff = np.sum(np.square(result - reference_result))
        raise AssertionError("Failure for function {:s},\n"
            "Sum of absolute diff to reference result {:f}\n"
            "Sum of squared diff to refence_result {:f}".format(name,
            diff, sum_of_squared_diff))