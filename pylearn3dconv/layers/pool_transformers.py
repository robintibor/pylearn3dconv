from pylearn3dconv.volumetric_space import Conv3DSpace
from pylearn3dconv.theanodnn.pool import dnn_pool3d2d

class PoolTransformer():
    op_axes = None # should be overwritten by subclass
    def __init__(self, pool_shape, pool_stride, image_shape, pool_type,
            input_axes):
        self.__dict__.update(locals())
        del self.self

    def pool(self, inputs):
        # maybe have to shuffle for pooling operation
        if self.input_axes != self.op_axes:
            inputs = Conv3DSpace.convert(inputs, self.input_axes, self.op_axes)
            
        rval = self.actual_pooling(inputs)

        if self.input_axes != self.op_axes:
            rval = Conv3DSpace.convert(rval, self.op_axes, self.input_axes)
        return rval
    
    
class CudnnPoolTransformer(PoolTransformer):
    op_axes = ('b', 'c', 0, 1, 2)
    
    def actual_pooling(self, inputs):
        # rename pool type for dnn ('mean' should be 'average')
        pool_type = self.pool_type
        if pool_type =='mean': 
            pool_type = 'average'
        return dnn_pool3d2d(inputs=inputs,
                           pool_shape=tuple(self.pool_shape),
                           pool_stride=tuple(self.pool_stride),
                           image_shape=tuple(self.image_shape),
                           mode=pool_type)
        