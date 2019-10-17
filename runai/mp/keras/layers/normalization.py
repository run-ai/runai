import keras.layers
import tensorflow as tf

import runai

from . import coordinator
from .parallelised import Parallelised

ATTRIBUTES = ['gamma', 'beta', 'moving_mean', 'moving_variance']

class BatchNormalization(Parallelised, keras.layers.BatchNormalization):
    """ A MP-supported implementation for keras.layers.BatchNormalization

    The implementation is highly coupled with the Keras version, and was
    implemented in regards of Keras 2.2.4.

    We rely on the fact that specified attributes are declared in build(),
    and are used only by call().

    Instead of implementing (copying Keras's code) the layer ourselves,
    we use the original implementation and cause it for the parallelisation.
    """
    def build(self, input_shape):
        def add_weight(name, shape, *args, **kwargs):
            """ This implementation will be called every time the original
            build() method will create a weight variable. Instead, we make
            it create multiple variables with smaller channel size and placed
            on different GPUs
            """
            dim = shape[0] // runai.mp.splits # all weights are of shape (c,)
            return self.add_weights(name, (dim,), *args, **kwargs)
        
        with runai.utils.Hook(self, 'add_weight', add_weight, recursion=False): # must not allow recursion as we call add_weight() ourselves
            super(BatchNormalization, self).build(input_shape)
        
        # every attribute declared by build() is actually a list of tensors,
        # and not a single one. we make sure they are not accessed by anyone
        # by renaming them
        for attribute in ATTRIBUTES:
            runai.utils.attribute.rename(self, attribute, '_' + attribute)
    
    def call(self, inputs, *args, **kwargs):
        assert not isinstance(inputs, (tuple, list))
        
        # we expect all inputs to have underlying parallelised tensors
        # we do not support the other cases as we have already built the
        # weights for the parallelised version
        assert coordinator.registered(inputs)

        runai.log.info('Using parallelised input for \'%s\' layer "%s"', self.__class__.__name__, getattr(self, 'name', 'N/A'))

        def output(gpu, input):
            values = [getattr(self, '_' + attribute) for attribute in ATTRIBUTES]
            values = [None if value is None else value[gpu] for value in values]
            
            with runai.utils.Attribute(self, ATTRIBUTES, values):
                with tf.device('/device:GPU:%d' % gpu):
                    return super(BatchNormalization, self).call(input, *args, **kwargs)

        outputs = [output(gpu, input) for gpu, input in enumerate(coordinator.resolve(inputs))]

        return self.merge(outputs, channel_axis=self.axis)
