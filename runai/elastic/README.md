# Run:AI Elasticity

A module for transforming static Keras and PyTorch models to be elastic

## Information

Elastic jobs automatically set their hyperparameters (e.g. batch size) according to the current available resources.
Elastic jobs use multi-GPU training and gradient accumulation (GA) in order to fully utilize the available resources.

## Getting Started

You can transform your Keras and PyTorch models to be elastic pretty easily.

### Prerequisites

As mentioned above, elastic jobs need use multi-GPU training as well as gradient accumulation.

#### GA
The [Run:AI Python library](../ga/README.md) provides GA-supported optimizers and wrappers (for generic optimizers) for Keras and PyTorch.

#### Multi-GPU
For multi-GPU training in Keras, [Horovod](https://github.com/horovod/horovod) is required.

In PyTorch, any multi-GPU mechanism could be used. We recommend using the [built-in](https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/data_parallel.py) `nn.DataParallel()` mechanism.

### Installing

Install the `runai` Python library using `pip` using the following command:

```
pip install runai
```

> Make sure to use the correct `pip` installer (you might need to use `pip3` for Python3)

## Usage

### Import The Package

First you need to import the respective package to your code.
Import the subpackage in respect to the framework you are using.

If you are using Keras, use the following line:

```
import runai.elastic.keras
```

If you are using PyTorch, use the following line:

```
import runai.elastic.torch
```

### Initialize The Module

After importing the respective Run:AI elasticity module, you will need to initialize it.
Upon initialization, you'll pass two main arguments:
1. *Global batch size* - The desired batch size. Using this batch size, the model might not fit into a single GPU - in terms of memory. The elasticity module will use GA and multi-GPU training to enable it.
2. *Maximum GPU batch size* - The maximum batch size that can be run in a single GPU in terms of memory. This will be used by the Run:AI elasticity module for determining whether to use GA or not, and for how many steps.

Call the `init()` method from the imported module and pass the two arguments. For example, if you are using PyTorch, use the following line:

```
runai.elastic.torch.init(256, 64)
```

This indicates that the desired global batch size is 256, but a maximum of batch size 64 can be fit into a single GPU.
The Run:AI elasticity module will determine - according to the available resources - what to do, to achieve the global batch size.
It will use multi-GPU training if there is more than a single GPU available, and it will use GA if there are not enough GPUs available.

The following table shows how many GA steps will be used for different scenarios with different number of available GPUs, for the given example:

| GPUs Available | GA Steps |
|----------------|----------|
| 1              | 4        |
| 2              | 2        |
| 4              | N/A      |


### Use The Module

#### Use The Calculated Parameters

After initialization, a number of parameters will be available at `runai.elastic`:

| Name | Value |
|------|-------|
| `steps` | Number of GA steps, or 1 if GA is not needed |
| `batch_size` | The batch size to be used in every step for every GPU |
| `global_batch_size` | The global batch size that was passed to `init()` |
| `gpus` | The number of GPUs that are used |
| `master` | Mainly relevant for Horovod. `True` if this is the master process (rank 0). |

When passing the batch size to any framework method, use `runai.elastic.batch_size`.

#### Keras

For Keras models, you'll need to wrap your `Model` object with `runai.elastic.keras.models.Model`. For example:

```
model = runai.elastic.keras.models.Model(model)
```

This will automatically wrap your optimizer with GA if necessary, and initialize Horovod.

#### PyTorch

For PyTorch models, you'll need to wrap your `Optimizer` with GA.
Refer to the [documentation](../ga/README.md) for more information.
Use `runai.elastic.steps` for the number of steps to be accumulated - the value for the `steps` argument of Run:AI GA optimizer. For example:

```
optimizer = runai.ga.torch.optim.Optimizer(optimizer, runai.elastic.steps)
```

Additionally, you'll need to data-parallelise your model.
As mentioned above, this could be done in any mechanism of your choice.
We recommend using the built-in `nn.DataParallel()` method. For example:

```
model = torch.nn.DataParallel(model)
```

## Examples

Examples of both Keras and PyTorch elastic jobs exist under the [examples](../../examples/elastic) directory:

### Keras

* [MNIST](../../examples/elastic/keras/mnist.py) - A simple example of a CNN model on MNIST
* [CIFAR10](../../examples/elastic/keras/cifar10.py) - Using the Keras builtin implementation of ResNet50 on the CIFAR10 dataset

> *NOTE:* The examples were tested using Python 3.6, Keras 2.2.4, TensorFlow 1.15.0

### PyTorch

* [MNIST](../../examples/elastic/torch/mnist.py) - A simple example of a CNN model on MNIST
* [CIFAR10](../../examples/elastic/torch/cifar10.py) - Using builtin models from `torchvision` on CIFAR10 dataset

> *NOTE:* The example was tested using Python 3.6, PyTorch 1.5.0
