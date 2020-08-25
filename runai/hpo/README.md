# Run:AI HPO Support

A helper library for hyperparameter optimization (HPO) experiments

## Getting Started

### Prerequisites

Run:AI HPO library is dependent on [PyYAML](https://github.com/yaml/pyyaml).
Install it using the command:

```
pip install pyyaml
```

### Installing

Install the `runai` Python library using `pip` using the following command:

```
pip install runai
```

> Make sure to use the correct `pip` installer (you might need to use `pip3` for Python3)

### Usage

* Import the ``runai.hpo`` package.

```
import runai.hpo
```

* Initialize the Run:AI HPO library with a path to a directory shared between all cluster nodes (typically using an NFS server).
We recommend specifying a unique name for the experiment, the name will be used to create a sub-directory on the shared folder. 

```
runai.hpo.init('/path/to/nfs', 'model-abcd-hpo')
```

* Decide on an HPO strategy:
    *  Random search - randomlly pick a set of hyperparameter values
    *  Grid search - pick the next set of hyperparameter values, iterating through all sets across multiple experiments

```
strategy = runai.hpo.Strategy.GridSearch
```


* Call the Run:AI HPO library to specify a set of hyperparameters and pick a specific configuration for this experiment.

```
config = runai.hpo.pick(
    grid=dict(
        batch_size=[32, 64, 128],
        lr=[1, 0.1, 0.01, 0.001]),
    strategy=strategy)
```

* Use the returned configuration in your code. For example:

```
optimizer = keras.optimizers.SGD(lr=config['lr'])
```

Metrics could be reported and saved in the experiment directory under runai.yaml using `runai.hpo.report`.
You should pass the epoch number and a dictionary with metrics to be reported. For example:
```
runai.hpo.report(epoch=5, metrics={ 'accuracy': 0.87 })
```

> This is an example with hard-coded values. In real life, you'll want to pass some other variables

### Examples

Examples are available [here](../../examples/hpo):
- [VGG16](../../examples/hpo/vgg16.py) - A simple usage of Run:AI HPO Support on a VGG16 model
