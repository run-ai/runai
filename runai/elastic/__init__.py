import sys

import runai.utils

# users should not call this method directly, but call the `init()` method of
# the sub-package of the respecive framework (e.g. `runai.elastic.keras.init()`)
def _init(global_batch_size, max_gpu_batch_size, gpus=None):
    if gpus is None:
        gpus = runai.utils.gpus.count()

    if gpus < 1:
        raise ValueError('GPU count (%d) must be at least 1' % gpus)

    module = sys.modules[__name__]

    setattr(module, 'global_batch_size', global_batch_size)
    setattr(module, 'gpus', gpus)
    setattr(module, 'master', True)

    # TODO(levosos): support uneven dividing
    steps = max(1, global_batch_size // (max_gpu_batch_size * gpus)) # must be at least 1
    batch_size = global_batch_size // (steps * gpus)

    setattr(module, 'steps', steps)
    setattr(module, 'batch_size', batch_size)

    runai.utils.log.info('Spreading global batch size %d across %d GPU(s) each with %d step(s) of batch size %d',
        global_batch_size, gpus, steps, batch_size)
