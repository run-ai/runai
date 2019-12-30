import unittest
from os import environ

import keras
from keras.utils.np_utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential
import keras.optimizers
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator

import runai.utils
import runai.reporter

NUM_CLASSES = 10
BATCH_SIZE = 16
STEPS_PER_EPOCH = 1

class MockReporter(runai.utils.Hook):
    def __init__(self, methodName):
        super(MockReporter, self).__init__(runai.reporter.keras_metric_reporter, methodName)
        self.reported = []

    def __hook__(self, *args, **kwargs):
        reportedInput = args[0]
        wasCurrentInputAlreadyReported = reportedInput not in self.reported
        if wasCurrentInputAlreadyReported:
            self.reported.append(reportedInput)

class KerasStopModelCallback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        self.model.stop_training = True

class KerasAutologTest(unittest.TestCase):
    def _run_test(self, run_fit=True, expected_metrics=[], expected_parameters=[]):
        self._mock_env_variables()

        with MockReporter('reportMetric') as reportMetricMock, MockReporter('reportParameter') as reportParameterMock:
            if run_fit:
                self._run_model_with_fit()
            else:
                self._run_model_with_fit_generator()

            self.assertEqual(reportMetricMock.reported, expected_metrics, 'Reported Metrics unmatched')
            self.assertEqual(reportParameterMock.reported, expected_parameters, 'Reported Paramters unmatched')

    def _mock_env_variables(self):
        environ["podUUID"] = "podUUId"
        environ["reporterGatewayURL"] = "reporterGatewayURL"

    def _run_model_with_fit(self):
        x_train, y_train = self._get_x_train_y_train()
        model = self._create_model_and_compile()

        model.fit(x_train, y_train, batch_size=BATCH_SIZE, callbacks=[
                  KerasStopModelCallback()])

    def _run_model_with_fit_generator(self):
        x_train, y_train = self._get_x_train_y_train()
        model = self._create_model_and_compile()
        datagen = ImageDataGenerator()
        datagen.fit(x_train)
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                            steps_per_epoch=STEPS_PER_EPOCH)

    def _create_model_and_compile(self):
        model = Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28, 1)))
        model.add(Dense(NUM_CLASSES, activation='softmax'))
        optimizer = keras.optimizers.Adam()

        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model


    def _get_x_train_y_train(self):
        (x_train, y_train), (_x_test, _y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 28, 28, 1)
        x_train = x_train.astype('float32')
        x_train /= 255
        y_train = to_categorical(y_train, NUM_CLASSES)
        return x_train, y_train

    def testFitWithoutAutoLog(self):
        self._run_test()

    def testFitWithAutoLog(self):
        runai.reporter.autolog()
        expected_metrics = ['overall_epochs', 'batch_size', 'number_of_layers', 'epoch', 'step', 'accuracy', 'loss']
        expected_parameters = ['optimizer_name', 'learning_rate']
        self._run_test(expected_metrics=expected_metrics, expected_parameters=expected_parameters)
        runai.reporter.disableAutoLog()

    def testFitAllMetrics(self):
        runai.reporter.autolog(loss_method=True, epsilon=True)
        expected_metrics = ['overall_epochs', 'batch_size', 'number_of_layers', 'epoch', 'step', 'accuracy', 'loss']
        expected_parameters = ['loss_method', 'optimizer_name', 'learning_rate', 'epsilon']
        self._run_test(expected_metrics=expected_metrics, expected_parameters=expected_parameters)
        runai.reporter.disableAutoLog()

    def testFitGeneratorWithoutAutoLog(self):
        self._run_test(run_fit=False)

    def testFitGeneratorWithAutoLog(self):
        runai.reporter.autolog()
        expected_metrics = ['overall_epochs', 'number_of_layers', 'epoch', 'step', 'accuracy', 'loss']
        expected_parameters = ['optimizer_name', 'learning_rate']
        self._run_test(run_fit=False, expected_metrics=expected_metrics, expected_parameters=expected_parameters)
        runai.reporter.disableAutoLog()

    def testFitGeneratorAllMetrics(self):
        runai.reporter.autolog(loss_method=True, epsilon=True)
        expected_metrics = ['overall_epochs', 'number_of_layers', 'epoch', 'step', 'accuracy', 'loss']
        expected_parameters = ['loss_method', 'optimizer_name', 'learning_rate', 'epsilon']
        self._run_test(run_fit=False, expected_metrics=expected_metrics, expected_parameters=expected_parameters)
        runai.reporter.disableAutoLog()

    #TODO: Add tests that will add new metrics to the compile method and verify they were added.

if __name__ == '__main__':
    unittest.main()
