# Run:AI Reporter Library

An interface to send metrics and parameters to Promethues Pushgateway.

### Installing

Install the `runai` Python library using `pip` using the following command:

```
pip install runai
```

> Make sure to use the correct `pip` installer (you might need to use `pip3` for Python3)

## Requirements

It is required to have the following environment variables:
1. `podUUID`
2. `reporterGatewayURL`

These environment variables will be added to each pod when a job was created using the Run:AI CLI.

### Reporting Options

*Parameters:*
Key-value input parameters of your choice. Key and value are strings.

*Metrics:*
Key-value input parameters of your choice. Key is a string, value is numeric.

## Usage

### Import

First you need to add the following import command to your code:

```
import runai.reporter
```

If you are using Keras, use the following command instead:

```
import runai.reporter.keras
```

> INFO: Keras related methods are in a sub-package to support non-Keras environments as well

### Scoped and Non-Scoped APIs

There are two ways to use the Run:AI reporting library.
The first approach is to use a reporter object as a Python context manager (i.e. using `with`).
The second approach is to call non-scoped methods.

The communication with the Pushgateway is done in another process, to reduce the performance impact of this library on the main process, making the reporting itself is done asynchronously.
Proper termination is required to ensure everything was successfully reported before the Python process terminates.

Therefore, it is **highly recommended to use the scoped API** as it shuts down the reporting process properly.

If using the non-scoped API, reports are not guaranteed to be successfully reported without explicit termination (using the `finish` method) or upon failures.

#### Scoped API - *Recommended*

First, you'll need to create a reporter object `runai.reporter.Reporter`.
You should create the object using a `with` statement. For example:

```
with runai.reporter.Reporter() as reporter:
    pass
```

Then, use `reporter` to report metrics and parameters using the methods `reportMetric` and `reportParameter`.

When using Keras, it is preferable to use `runai.reporter.keras.Reporter` instead.
The Keras reporter lets you to add automatic logging to Keras models.

This could be done by passing `autolog=True` when creating the Keras reporter, or by calling its `autolog()` method. For example:

```
with runai.reporter.keras.Reporter(autolog=True) as reporter:
    pass
```

or

```
with runai.reporter.keras.Reporter() as reporter:
    reporter.autolog()
```

#### Non-Scoped API

*It is less recommended to use this API (use the scoped API instead)*

Use the methods `reportMetric` and `reportParameter` from `runai.reporter` directly without creating any reporter objects.

> INFO: This actually creates a global `Reporter` object in the background

At the end of your script, you should explicitly call `runai.reporter.finish()`.
This is to properly terminate the reporter process and ensure everything was successfully reported before the Python script ends.

### API Reference

---

`runai.reporter.Reporter.reportMetric` and `runai.reporter.reportMetric`

Sends a metric with the following name "reporter_push_gateway_metric_[reporter_metric_name]".

Scoped API:
```
with runai.reporter.Reporter() as reporter:
    reporter.reportMetric('batch_size', 100)
```

Non-scoped API:
```
runai.reporter.reportMetric('batch_size', 100)
```

---

`runai.reporter.Reporter.reportParameter` and `runai.reporter.reportParameter`

Sends a parameter with the following name "reporter_push_gateway_metric_[reporter_parameter_name]".

Scoped API:
```
with runai.reporter.Reporter() as reporter:
    reporter.reportParameter('loss_method', 'categorical_crossentropy')
```

Non-scoped API:
```
runai.reporter.reportParameter('loss_method', 'categorical_crossentropy')
```

### Scoped API

---

`runai.reporter.keras.Reporter.autolog`

Enables Keras automatic logging. This could be done by passing `autolog=True` when creating the Keras reporter, or by calling its `autolog()` method. For example:

```
with runai.reporter.keras.Reporter(autolog=True) as reporter:
    pass
```

or

```
with runai.reporter.keras.Reporter() as reporter:
    reporter.autolog()
```

### Non-Scoped API

---

`runai.reporter.finish`

Make sure all reports were sent successfully and gracefully terminate the background process (used for communication with Pushgateway).
Simply add the following line at the very end of your script:

```
runai.reporter.finish()
```

---

`runai.reporter.keras.autolog`

Enables automatic metrics and parameters updates for Keras.
Simply add the following line before running the model:

```
runai.reporter.keras.autolog()
```

---

`runai.reporter.keras.disableAutolog`

Disables automatic metrics and parameters updates from Keras fit, fit_generator methods.
Simply add the following line before running the model

```
runai.reporter.keras.disableAutolog()
```
