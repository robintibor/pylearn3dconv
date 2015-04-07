def generate__theano_test_data(rng):
    from theano import shared
    inputs_val,filters_val, bias_val = generate_test_data(rng)
    inputs = shared(inputs_val)
    filters = shared(filters_val)
    #TODOREMOVE?:bias = shared(np.zeros(filters_shape[0]).astype('float32'))
    bias = shared(bias_val)
    return inputs, filters, bias


def generate_test_data(rng):
    inputs_shape=(16, 20, 32, 16, 1)
    filters_shape=(10, 6, 12, 4, 1)
    inputs_shape=(10, 12, 17, 13, 1)
    filters_shape=(10, 6, 12, 4, 1)
    inputs_shape=(8, 10, 16, 8, 1)
    filters_shape=(5, 3, 6, 2, 1)
    #inputs_shape=(4, 3, 5, 7, 2)
    #filters_shape=(3, 2, 3, 4, 2)
    #inputs_shape=(3, 2, 4, 5, 2)
    #filters_shape=(1, 2, 3, 2, 2)
    #inputs_shape=(2, 1, 1, 1, 1)
    #filters_shape=(2, 1, 1, 1, 1)
    inputs = rng.normal(size=inputs_shape).astype('float32')
    filters = rng.normal(size=filters_shape).astype('float32')
    bias = rng.normal(size=filters_shape[0]).astype('float32')
    return inputs, filters, bias