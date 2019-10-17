from __future__ import absolute_import # to support 'import keras.backend' in Python 2

import enum

class Method(enum.Enum):
    Cin  = 0
    Cout = 1

from . import keras

from runai.utils import log

def init(splits, method):
    if method != Method.Cin and method != Method.Cout:
        raise ValueError('Unrecognized method: %s' % method)

    log.info('Initializing MP (%s) with %s splits', method.name, splits)
    
    import sys

    setattr(sys.modules[__name__], 'splits', splits)
    setattr(sys.modules[__name__], 'method', method)

    import keras.layers

    from .keras import layers

    keras.layers.Activation         = layers.Activation
    keras.layers.Add                = layers.Add
    keras.layers.Average            = layers.Average
    keras.layers.BatchNormalization = layers.BatchNormalization
    keras.layers.Conv2D             = layers.Conv2D
    keras.layers.Dense              = layers.Dense
    keras.layers.Dropout            = layers.Dropout
    keras.layers.Maximum            = layers.Maximum
    keras.layers.MaxPooling2D       = layers.MaxPooling2D
    keras.layers.Minimum            = layers.Minimum
    keras.layers.Multiply           = layers.Multiply
    keras.layers.Subtract           = layers.Subtract
    keras.layers.ZeroPadding2D      = layers.ZeroPadding2D
