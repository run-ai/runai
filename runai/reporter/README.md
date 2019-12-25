# Run:AI Reporter Library

An interface to send metrics and parameters to Promethues Push Gateway.

## Concepts
### Parameters
Key-value input parameters of your choice. Key and value are strings.

### Metrics
Key-value input parameters of your choice. Key is a string, value is numeric.

## Getting Started

Follow the next instructions to use the Reporter library.

### Installing

Install the `runai` Python library using `pip` using the following command:

```
pip install runai
```

> Make sure to use the correct `pip` installer (you might need to use `pip3` for Python3)

## Requirements
It is required to have the following environment variables:
1) `podUUID`
2) `reporterGatewayURL`

These environment variables will be added to each pod when a job was created by `arena runai` commands.

## Usage

First you need to add the following import command to your code:

```
import runai.reporter
```

### Methods
#### reportMetric

Sends a metric with the following name "reporter_push_gateway_metric_[reporter_metric_name]".
##### usage example:
```
runai.reporter.reportMetric('batch_size', 100)
```

#### reportParameter

Sends a parameter with the following name "reporter_push_gateway_metric_[reporter_parameter_name]".
##### usage example:
```
runai.reporter.reportParameter('loss_method', 'categorical_crossentropy')
```

#### autolog

Enables automatic metrics and parameters updates for Keras.
##### usage:
Simply add the following line before running the model
```
runai.reporter.autolog()
```


#### disableAutolog

Disables automatic metrics and parameters updates from Keras fit, fit_generator methods.
##### usage:
Simply add the following line before running the model
```
runai.reporter.disableAutolog()
```
