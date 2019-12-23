import time
from runai.reporterlib import reportMetric, reportParameter
import keras

__original_fit__ = keras.Model.fit
__original_compile__ = keras.Model.compile
__original_fit_generator__ = keras.Model.fit_generator

def disableAutoLog():
    keras.Model.fit = __original_fit__
    keras.Model.fit_generator = __original_fit_generator__
    keras.Model.compile = __original_compile__

def autolog(acc = True, loss = True, learning_rate = True, epoch = True, step = True, batch_size = True, overall_epochs = True,
            optimizer_name = True, num_layers = True, loss_method = False, epsilon = False):
    inputs = locals()
    class KerasAutoMetricReporter(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            reportParameterIfNeeded(inputs, 'optimizer_name', type(self.model.optimizer).__name__)
            reportMetricIfNeeded(inputs, 'num_layers', len(self.model.layers))

            if hasattr(self.model.optimizer, 'lr'):
                lr = self.model.optimizer.lr if \
                    type(self.model.optimizer.lr) is float \
                    else keras.backend.eval(self.model.optimizer.lr)
                reportParameterIfNeeded(inputs, 'learning_rate', lr)

            if epsilon and hasattr(self.model.optimizer, 'epsilon'):
                epsilon_val = self.model.optimizer.epsilon if \
                    type(self.model.optimizer.epsilon) is float \
                    else keras.backend.eval(self.model.optimizer.epsilon)
                reportParameter('epsilon', epsilon_val)

        def on_batch_end(self, batch, logs={}):
            reportMetricIfNeeded(inputs, 'step', batch)

            reportMetricFromLogs(inputs, "acc", logs)
            reportMetricFromLogs(inputs,"loss", logs)

        def on_epoch_begin(self, epoch_val, logs=None):
            reportMetric('epoch', epoch_val)

        def on_epoch_end(self, epoch_val, logs=None):
            reportMetric('epoch', epoch_val)
            if hasattr(self.model.optimizer, 'lr'):
                lr = self.model.optimizer.lr if \
                    type(self.model.optimizer.lr) is float \
                    else keras.backend.eval(self.model.optimizer.lr)
                reportParameterIfNeeded(inputs, 'learning_rate', lr)

    def fit(self, *args, **kwargs):
        tmp_list = list(args)

        if len(tmp_list) >= 5:
            tmp_list[5] += [KerasAutoMetricReporter()]
        elif 'callbacks' in kwargs:
            kwargs['callbacks'] += [KerasAutoMetricReporter()]
        else:
            kwargs['callbacks'] = [KerasAutoMetricReporter()]

        overall_number_of_epochs = 1
        if len(tmp_list) >= 4:
            overall_number_of_epochs = tmp_list[4]
        elif ('epochs' in kwargs):
            overall_number_of_epochs = kwargs['epochs']
        reportMetricIfNeeded(inputs, 'overall_epochs', overall_number_of_epochs)

        batch_size_val = ""
        if len(tmp_list) >= 3:
            batch_size_val = tmp_list[3]
        elif ('batch_size' in kwargs):
            batch_size_val = kwargs['batch_size']
            print("yodar batch size:", batch_size_val)
        if batch_size_val:
            reportMetricIfNeeded(inputs, 'batch_size', batch_size_val)

        args = tuple(tmp_list)
        return __original_fit__(self, *args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        tmp_list = list(args)

        if len(tmp_list) >= 5:
            tmp_list[5] += [KerasAutoMetricReporter()]
        elif 'callbacks' in kwargs:
            kwargs['callbacks'] += [KerasAutoMetricReporter()]
        else:
            kwargs['callbacks'] = [KerasAutoMetricReporter()]

        overall_number_of_epochs = 1
        if len(tmp_list) >= 3:
            overall_number_of_epochs = tmp_list[3]
        elif ('epochs' in kwargs):
            overall_number_of_epochs = kwargs['epochs']
        if overall_number_of_epochs:
            reportMetricIfNeeded(inputs, 'overall_epochs', overall_number_of_epochs)

        args = tuple(tmp_list)

        return __original_fit_generator__(self, *args, **kwargs)

    def compile(self, *args, **kwargs):
        tmp_list = list(args)
        metrics_to_add = ['acc']
        if len(tmp_list) >= 3:
            for metric in metrics_to_add:
                if not metric in tmp_list[3]:
                    tmp_list[3].append(metric)
        elif 'metrics' in kwargs:
            for metric in metrics_to_add:
                if not metric in kwargs['metrics']:
                    kwargs['metrics'].append(metric)
        else:
            kwargs['metrics'] = metrics_to_add

        loss_method_val = ""
        if len(tmp_list) >= 2:
            loss_method_val = args[2]
        elif 'loss' in kwargs:
            loss_method_val = kwargs['loss']
        if loss_method_val:
            reportParameterIfNeeded(inputs, 'loss_method', loss_method_val)

        args = tuple(tmp_list)
        return __original_compile__(self, *args, **kwargs)


    keras.Model.fit = fit
    keras.Model.fit_generator = fit_generator
    keras.Model.compile = compile

def reportMetricFromLogs(inputs, key, logs):
    if key in logs and key in inputs and inputs[key]:
            reportMetric(key, logs[key])

def reportMetricIfNeeded(inputs, key, value):
    if key in inputs and inputs[key]:
        reportMetric(key, value)

def reportParameterIfNeeded(inputs, key, value):
    print("inputs: ", inputs)
    if key in inputs and inputs[key]:
        reportParameter(key, value)
