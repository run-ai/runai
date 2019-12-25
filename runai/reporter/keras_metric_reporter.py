import time
from runai.reporter import reportMetric, reportParameter
import keras

__original_fit__ = keras.Model.fit
__original_compile__ = keras.Model.compile
__original_fit_generator__ = keras.Model.fit_generator

def disableAutoLog():
    keras.Model.fit = __original_fit__
    keras.Model.fit_generator = __original_fit_generator__
    keras.Model.compile = __original_compile__

def autolog(acc=True, loss=True, learning_rate=True, epoch=True, step=True, batch_size=True, overall_epochs=True,
            optimizer_name=True, num_layers=True, loss_method=False, epsilon=False):
    # The following line must be at top of the method
    autolog_inputs = locals()

    def fit(self, *args, **kwargs):
        callbacks_index_in_args = 5
        callbacks_value_to_add = [KerasAutoMetricReporter()]
        _add_value_to_method_parameter(args, kwargs, callbacks_index_in_args, 'callbacks', callbacks_value_to_add)
        _add_value_to_method_parameter(args, kwargs, callbacks_index_in_args, 'callbacks', callbacks_value_to_add)

        if _should_report_metric_or_parameter(autolog_inputs, 'overall_epochs'):
            keras_fit_default_number_of_epochs = 1
            epochs_index_args = 3
            overall_epochs_val = _get_value_of_method_parameter(args, kwargs, epochs_index_args, 'epochs')
            if overall_epochs_val:
                reportMetric('overall_epochs', overall_epochs_val)
            else:
                reportMetric('overall_epochs', keras_fit_default_number_of_epochs)

        if _should_report_metric_or_parameter(autolog_inputs, 'batch_size'):
            batch_size_index_args = 2
            batch_size_val = _get_value_of_method_parameter(args, kwargs, batch_size_index_args, 'batch_size')
            if batch_size_val:
                reportMetric('batch_size', batch_size_val)

        return __original_fit__(self, *args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        callbacks_index_in_args = 4
        callbacks_value_to_add = [KerasAutoMetricReporter()]
        _add_value_to_method_parameter(args, kwargs, callbacks_index_in_args, 'callbacks', callbacks_value_to_add)
        _add_value_to_method_parameter(args, kwargs, callbacks_index_in_args, 'callbacks', callbacks_value_to_add)

        if _should_report_metric_or_parameter(autolog_inputs, 'overall_epochs'):
            keras_fit_generator_default_number_of_epochs = 1
            epochs_index_args = 2
            overall_epochs_val = _get_value_of_method_parameter(args, kwargs, epochs_index_args, 'epochs')
            if overall_epochs_val:
                reportMetric('overall_epochs', overall_epochs_val)
            else:
                reportMetric('overall_epochs', keras_fit_generator_default_number_of_epochs)

        return __original_fit_generator__(self, *args, **kwargs)

    def compile(self, *args, **kwargs):
        _add_auto_log_metrics(args, kwargs)
        loss_method_index_args = 1
        loss_method_val = _get_value_of_method_parameter(args, kwargs, loss_method_index_args, 'loss')
        if loss_method_val:
            _report_parameter_if_needed(autolog_inputs, 'loss_method', loss_method_val)

        return __original_compile__(self, *args, **kwargs)

    keras.Model.fit = fit
    keras.Model.fit_generator = fit_generator
    keras.Model.compile = compile

    def _add_value_to_method_parameter(original_args, original_kwargs, parameter_args_index, parameter_kwargs_key, value):
        if parameter_kwargs_key in original_kwargs:
            original_kwargs[parameter_kwargs_key] += value
        elif parameter_args_index < len(original_args):
            original_args[parameter_args_index] += value
        else:
            original_kwargs[parameter_kwargs_key] = value

    def _get_value_of_method_parameter(original_args, original_kwargs, parameter_args_index, parameter_kwargs_key):
        if parameter_kwargs_key in original_kwargs:
            return original_kwargs[parameter_kwargs_key]
        elif parameter_args_index < len(original_args):
            return original_args[parameter_args_index]
        return None

    def _add_auto_log_metrics(original_args, original_kwargs):
        metrics_args_index = 2
        metrics_to_add = ['acc']
        if 'metrics' in original_kwargs:
            _add_metric_if_not_exist(original_kwargs['metrics'], metrics_to_add)
        elif len(original_args) > metrics_args_index:
            _add_metric_if_not_exist(original_args[metrics_args_index], metrics_to_add)
        else:
            original_kwargs['metrics'] = metrics_to_add

    def _add_metric_if_not_exist(metric_list, metrics_to_add):
        for metric in metrics_to_add:
            if not metric in metric_list:
                metric_list.append(metric)

    def _report_metric_from_logs_if_needed(autolog_inputs, key, logs):
        does_key_exist_in_log = key in logs
        if does_key_exist_in_log and _should_report_metric_or_parameter(autolog_inputs, key):
            reportMetric(key, logs[key])

    def _report_metric_if_needed(autolog_inputs, key, value):
        if _should_report_metric_or_parameter(autolog_inputs, key):
            reportMetric(key, value)

    def _report_parameter_if_needed(autolog_inputs, key, value):
        if _should_report_metric_or_parameter(autolog_inputs, key):
            reportParameter(key, value)

    def _should_report_metric_or_parameter(autolog_inputs, key):
        return key in autolog_inputs and autolog_inputs[key]

    class KerasAutoMetricReporter(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            _report_parameter_if_needed(autolog_inputs, 'optimizer_name', type(
                self.model.optimizer).__name__)
            _report_metric_if_needed(
                autolog_inputs, 'num_layers', len(self.model.layers))
            self._handle_learning_rate_report()
            self._handle_epsilon_report()

        def on_batch_end(self, batch, logs={}):
            _report_metric_if_needed(autolog_inputs, 'step', batch)
            _report_metric_from_logs_if_needed(autolog_inputs, "acc", logs)
            _report_metric_from_logs_if_needed(autolog_inputs, "loss", logs)

        def on_epoch_begin(self, epoch_val, logs=None):
            _report_metric_if_needed(autolog_inputs, 'epoch', epoch_val)

        def on_epoch_end(self, epoch_val, logs=None):
            _report_metric_if_needed(autolog_inputs, 'epoch', epoch_val)
            self._handle_learning_rate_report()

        def _handle_learning_rate_report(self):
            if not _should_report_metric_or_parameter(autolog_inputs, 'learning_rate') or not hasattr(self.model.optimizer, 'lr'):
                return

            lr = self.model.optimizer.lr if \
                type(self.model.optimizer.lr) is float \
                else keras.backend.eval(self.model.optimizer.lr)
            reportParameter('learning_rate', lr)

        def _handle_epsilon_report(self):
            if not _should_report_metric_or_parameter(autolog_inputs, 'epsilon') or not hasattr(self.model.optimizer, 'epsilon'):
                return

            epsilon_val = self.model.optimizer.epsilon if \
                type(self.model.optimizer.epsilon) is float \
                else keras.backend.eval(self.model.optimizer.epsilon)
            reportParameter('epsilon', epsilon_val)
