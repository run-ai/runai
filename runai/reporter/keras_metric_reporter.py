import time

import keras

from runai.reporter import reportMetric, reportParameter

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
        _append_autolog_metrics_to_callbacks(args, kwargs, index_in_args=5)
        _report_overall_epochs(args, kwargs, index_in_args=3)
        _report_batch_size(args, kwargs)

        return __original_fit__(self, *args, **kwargs)

    def fit_generator(self, *args, **kwargs):
        _append_autolog_metrics_to_callbacks(args, kwargs, index_in_args=4)
        _report_overall_epochs(args, kwargs, index_in_args=2)

        return __original_fit_generator__(self, *args, **kwargs)

    def compile(self, *args, **kwargs):
        _add_autolog_metrics(args, kwargs)
        _report_loss_method(args, kwargs)
        _report_loss_method(args, kwargs)

        return __original_compile__(self, *args, **kwargs)

    keras.Model.fit = fit
    keras.Model.fit_generator = fit_generator
    keras.Model.compile = compile


    def _report_overall_epochs(original_args, original_kwargs, index_in_args=None):
        if index_in_args is None:
            raise ValueError("'index_in_args' must be specified")

        if _should_report_metric_or_parameter(autolog_inputs, 'overall_epochs'):
            overall_epochs_val = _get_value_of_method_parameter(original_args, original_kwargs, index_in_args, key_in_kwargs='epochs', default_value=1)  # Keras' default value
            reportMetric('overall_epochs', overall_epochs_val)

    def _report_batch_size(original_args, original_kwargs):
        if _should_report_metric_or_parameter(autolog_inputs, 'batch_size'):
            batch_size_val = _get_value_of_method_parameter(original_args, original_kwargs, index_in_args=2, key_in_kwargs='batch_size')
            if batch_size_val:
                reportMetric('batch_size', batch_size_val)

    def _report_loss_method(original_args, original_kwargs):
        loss_method_val = _get_value_of_method_parameter(original_args, original_kwargs, index_in_args=1, key_in_kwargs='loss')
        if loss_method_val:
            _report_parameter_if_needed(autolog_inputs, 'loss_method', loss_method_val)

    def _append_autolog_metrics_to_callbacks(original_args, original_kwargs, index_in_args=None):
        if index_in_args is None:
            raise ValueError("'index_in_args' must be specified")

        if 'callbacks' in original_kwargs:
            original_kwargs['callbacks'].append(KerasAutoMetricReporter())
        elif index_in_args < len(original_args):
            original_args[index_in_args].append(KerasAutoMetricReporter())
        else:
            original_kwargs['callbacks'] = [KerasAutoMetricReporter()]

    def _get_value_of_method_parameter(original_args, original_kwargs, index_in_args=None, key_in_kwargs=None, default_value=None):
        if index_in_args is None:
            raise ValueError("'index_in_args' must be specified")

        if key_in_kwargs is None:
            raise ValueError("'key_in_kwargs' must be specified")

        if key_in_kwargs in original_kwargs:
            return original_kwargs[key_in_kwargs]
        elif index_in_args < len(original_args):
            return original_args[index_in_args]
        return default_value

    def _add_autolog_metrics(original_args, original_kwargs):
        metrics_index_in_args = 2
        if len(original_args) > metrics_index_in_args:
            metrics = original_args[metrics_index_in_args]
        else:
            if 'metrics' not in original_kwargs:
                original_kwargs['metrics'] = []

            metrics = original_kwargs['metrics']

        autolog_metrics_from_logs = ['acc']
        for metric in autolog_metrics_from_logs:
            if metric not in metrics:
                metrics.append(metric)

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
            _report_parameter_if_needed(autolog_inputs, 'optimizer_name', type(self.model.optimizer).__name__)
            _report_metric_if_needed(autolog_inputs, 'num_layers', len(self.model.layers))

            self._report_parameter_from_model_optimizer('learning_rate', 'lr')
            self._report_parameter_from_model_optimizer('epsilon')

        def on_batch_end(self, batch, logs={}):
            _report_metric_if_needed(autolog_inputs, 'step', batch)
            self._report_metric_from_logs_if_needed(autolog_inputs, "acc", logs)
            self._report_metric_from_logs_if_needed(autolog_inputs, "loss", logs)

        def on_epoch_begin(self, epoch_val, logs=None):
            _report_metric_if_needed(autolog_inputs, 'epoch', epoch_val)

        def on_epoch_end(self, epoch_val, logs=None):
            _report_metric_if_needed(autolog_inputs, 'epoch', epoch_val)
            self._report_parameter_from_model_optimizer('learning_rate', 'lr')

        def _report_parameter_from_model_optimizer(self, metric_name, name_of_optimizer_attr=None):
            if not name_of_optimizer_attr:
                name_of_optimizer_attr = metric_name

            if not _should_report_metric_or_parameter(autolog_inputs, metric_name) or not hasattr(self.model.optimizer, name_of_optimizer_attr):
                return

            parameter_from_model = getattr(self.model.optimizer, name_of_optimizer_attr)
            value = parameter_from_model if type(parameter_from_model) is float else keras.backend.eval(parameter_from_model)
            reportParameter(metric_name, value)

        def _report_metric_from_logs_if_needed(self, autolog_inputs, key, logs):
            if key in logs:
                _report_metric_if_needed(autolog_inputs, key, logs[key])
