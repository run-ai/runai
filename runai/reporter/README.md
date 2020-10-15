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

## Usage

### Import

First you need to add the following import command to your code:

```
import runai.reporter
```

If you are using Keras, you may also want to import the following command for auto-logging:

```
import runai.reporter.keras
```

> INFO: Keras related methods are in a sub-package to support non-Keras environments as well

### Termination

The communication with the Pushgateway is done in another process, to reduce the performance impact of this library on the main process.

Therefore, at the end of your script, it is highly recommended to call `runai.reporter.finish()`.
This method makes sure all reports were successfully sent and gracefully terminates the worker process.

### Concepts

*Parameters:*
Key-value input parameters of your choice. Key and value are strings.

*Metrics:*
Key-value input parameters of your choice. Key is a string, value is numeric.

### API

---

`runai.reporter.reportMetric`

Sends a metric with the following name "reporter_push_gateway_metric_[reporter_metric_name]".

Example:
```
runai.reporter.reportMetric('batch_size', 100)
```

---

`runai.reporter.reportParameter`

Sends a parameter with the following name "reporter_push_gateway_metric_[reporter_parameter_name]".

Example:
```
runai.reporter.reportParameter('loss_method', 'categorical_crossentropy')
```

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
