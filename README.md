# Run:AI Python library

Public functional modules for Keras and TF

## Info

### Status

CircleCI is used for CI system:
[![CircleCI](https://circleci.com/gh/run-ai/runai/tree/master.svg?style=svg&circle-token=438045a8ae6e2d2a2253bae97ccf82dc94bcfd5b)](https://circleci.com/gh/run-ai/runai/tree/master)

### Modules

This library consists of a few pretty much independent submodules:

| Module | Name | Info |
|--------|------|------|
| Elastic | `elastic` | Make Keras models elastic |
| Gradient Accumulation | `ga` | [Gradient accumulation for Keras optimizers](runai/ga/README.md) |
| Model Parallelism | `mp` | Model-parallelism support for Keras builtin layers |
| Auto Profiler | `profiler` | Export timeline of TF/Keras models easily |
| Reporter | `reporter` | [An interface to send metrics and parameters to Promethues Push Gateway](runai/reporter/README.md) |

## Getting Started

### Installing

Install the `runai` Python library using `pip` using the following command:

```
pip install runai
```

> Make sure to use the correct `pip` installer (you might need to use `pip3` for Python3)

### Running The Tests

All tests (unit tests) can be run using the following command:

```
python -m unittest discover -s tests -v
```
