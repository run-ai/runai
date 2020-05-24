from __future__ import absolute_import # to support 'import keras.backend' in Python 2

import runai.elastic
import runai.utils

from . import models

def init(global_batch_size, max_gpu_batch_size, gpus=None):
    # first of all calculate the number of GA steps and the batch size
    runai.elastic._init(global_batch_size, max_gpu_batch_size, gpus)

    # now use Horovod if needed
    if runai.elastic.gpus > 1:
        runai.utils.log.debug('Initializing Horovod')
        import horovod.keras as hvd
        hvd.init()

        setattr(runai.elastic, 'master', hvd.local_rank() == 0)
        setattr(runai.elastic, 'hvd', hvd) # so that anyone will be easily accessible to Horovod

        runai.utils.log.debug('Attaching Keras session to GPU #%d', hvd.local_rank())
        import tensorflow
        config = tensorflow.ConfigProto()
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        import keras.backend
        keras.backend.set_session(tensorflow.Session(config=config)) # TODO(levosos): support cases where configuration will be set afterwards
