# Run:AI Gradient Accumulation

A generic gradient accumulation wrapper for Keras and PyTorch optimizers

## References

We published a series of articles on Medium related to gradient accumulation. This series of articles explains potential issues when using large batch sizes, what is gradient accumulation, how it works, and how we implemented it:

* [The problem of batch sizing and limited GPU memory](https://towardsdatascience.com/how-to-break-gpu-memory-boundaries-even-with-large-batch-sizes-7a9c27a400ce?source=friends_link&sk=74a7a2793da909c1194c0add818c7fd3)
* [What is Gradient Accumulation and how does it help?](https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa?source=friends_link&sk=28226e1d0ffa7e450d7dffa8d5b9cff6)
* [How-to guide to using the gradient accumulation mechanism and how we implemented it](https://towardsdatascience.com/how-to-easily-use-gradient-accumulation-in-keras-models-fa02c0342b60?source=friends_link&sk=ff9137c1c7fa5bbfc4c4e09bacc0273b)

## Getting Started

Follow the next instructions to use gradient accumulation in your own Keras and PyTorch models, or with one of our examples of basic models.

### Installing

Install the `runai` Python library using `pip` using the following command:

```
pip install runai
```

> Make sure to use the correct `pip` installer (you might need to use `pip3` for Python3)

### Usage

First you need to import the respective package to your code.
Import the subpackage in respect to the framework you are using.

If you are using Keras, use the following line:

```
import runai.ga.keras
```

If you are using PyTorch, use the following line:

```
import runai.ga.torch
```

Then, you need to choose one of the two possible ways to use gradient accumulation:

#### Wrap an existing optimizer with gradient accumulation

In case you have an instance of a Keras or PyTorch optimizer, especially if you have implemented an optimizer by yourself, you should wrap your `optimizer` with GA.

If you are using Keras, use the following line:

```
optimizer = runai.ga.keras.optimizers.Optimizer(optimizer, steps=STEPS)
```

If you are using PyTorch, use the following line:

```
optimizer = runai.ga.torch.optim.Optimizer(optimizer, steps=STEPS)
```

#### Create a gradient-accumulated common optimizer

You can create an instance of any common optimizer, already wrapped with gradient accumulation, out of the box.

Just create an instance of the your selected optimizer from the Run:AI library.
Import the optimizer either from `runai.ga.keras.optimizers` or from `runai.ga.torch.optim`.

For example, to create an `Adam` Keras optimizer, wrapped with gradient accumulation, you can use the following command:

```
optimizer = runai.ga.keras.optimizers.Adam(steps=STEPS)
```

Both ways require an argument `steps` to indicate the number of steps to accumulate gradients over (in the code above we are accumulating the gradients of `STEPS` steps).

> *NOTE:* It is not mandatory to pass `steps` as a keyword argument to the creation of the instance, and the `steps=` prefix may be removed

## Examples

Examples of wrappring Keras optimizers with gradient accumulation exist under the [examples](../../examples/ga) directory:

### Keras

* [VGG16](../../examples/ga/keras/vgg16.py) - **Recommended** - A very lean and simple example using the Keras builtin implementation of VGG16 on the CIFAR10 dataset
* [MLP](../../examples/ga/keras/mlp.py) - A very simple implementation of an MLP network on the MNIST dataset

> *NOTE:* The examples were tested using Python 3.6, Keras 2.2.4, TensorFlow 1.15.0

### PyTorch

* [MNIST](../../examples/ga/torch/mnist.py) - **Recommended** - A simple example of a CNN model on MNIST

> *NOTE:* The example was tested using Python 3.6, PyTorch 1.5.0
