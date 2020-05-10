import random
import unittest

import keras.backend as K
import keras.optimizers
import numpy as np

import runai.ga.keras

class Optimizer(unittest.TestCase):
    def _run(self, params, gradients, optimizer):
        params = [K.variable(param) for param in params]
        placeholders = [K.placeholder(shape=params[i].shape) for i in range(len(params))]

        with runai.ga.keras.hooks.get_gradients(optimizer, placeholders):
            updates = optimizer.get_updates(None, params) # 'loss' argument is unnecessary because of our hook

        for step in range(len(gradients)):
            K.get_session().run(
                updates,
                feed_dict={ placeholders[i]: gradients[step][i] for i in range(len(params)) }
            )

        return K.get_session().run(params)

    def _test(self, name):
        count = random.randint(2, 10)
        steps = random.randint(2, 10)
        shape = [random.randint(2, 5) for _ in range(2, 5)]

        # numpy generates float64 values by default, whereas TF uses float32 by default
        # we explictly generate float32 values because passing 64-bit floats to TF causes floating-point issues
        params = [np.random.random(shape).astype(np.float32) for _ in range(count)]
        gradients = [[np.random.random(shape).astype(np.float32) for _ in range(count)] for _ in range(steps)]

        them = self._run(
            params,
            [[np.array([gradients[step][i] for step in range(steps)]).sum(axis=0) / steps for i in range(count)]], # running a single step with the reduced (sum) gradients
            getattr(keras.optimizers, name)()
        )

        wrapped = runai.ga.keras.optimizers.Optimizer(getattr(keras.optimizers, name)(), steps=steps)
        self.assertEqual(wrapped.__class__.__name__, 'Optimizer')

        specific = getattr(runai.ga.keras.optimizers, name)(steps=steps)
        self.assertEqual(specific.__class__.__name__, name)

        for optimizer in [wrapped, specific]:
            self.assertEqual(optimizer.optimizer.__class__.__name__, name) # optimizer.optimizer is the Keras optimizer

            us = self._run(
                params,
                gradients,
                optimizer
            )

            self.assertTrue(np.allclose(us, them))

    def testSGD(self):
        self._test('SGD')

    def testRMSprop(self):
        self._test('RMSprop')

    def testAdagrad(self):
        self._test('Adagrad')

    def testAdadelta(self):
        self._test('Adadelta')

    def testAdam(self):
        self._test('Adam')

    def testAdamax(self):
        self._test('Adamax')

    def testNadam(self):
        self._test('Nadam')

if __name__ == '__main__':
    unittest.main()
