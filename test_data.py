from theano import shared
def generate_theano_test_data(rng,inputs_shape=(8, 10, 16, 8, 1),
    filters_shape=(5, 3, 6, 2, 1)):
    inputs_val,filters_val, bias_val = generate_test_data(rng,
        inputs_shape, filters_shape)
    inputs = shared(inputs_val)
    filters = shared(filters_val)
    #TODOREMOVE?:bias = shared(np.zeros(filters_shape[0]).astype('float32'))
    bias = shared(bias_val)
    return inputs, filters, bias


def generate_test_data(rng, inputs_shape=(8, 10, 16, 8, 1),
    filters_shape=(5, 3, 6, 2, 1)):
    inputs = rng.normal(size=inputs_shape).astype('float32')
    filters = rng.normal(size=filters_shape).astype('float32')
    bias = rng.normal(size=filters_shape[0]).astype('float32')
    return inputs, filters, bias