import sys

import torch.optim

import runai.utils

class State:
    def __init__(self, steps):
        self.steps = steps
        self.step = 0

# this design is inspired by Horovod's `DistributedOptimizer`
def Optimizer(optimizer, steps):
    """
    Wraps any valid PyTorch optimizer with gradient accumulation
    """
    class _GradientAccumulationOptimizer(torch.optim.Optimizer):
        def __init__(self, steps, params):
            self.__class__.__optimizer__.__init__(self, params) # TODO(levosos): what about `defaults`?
            self.state['runai'] = State(steps)

            runai.utils.log.debug('Wrapping \'%s\' PyTorch optimizer with GA of %d steps', optimizer.__class__.__name__, steps)

        def step(self, closure=None):
            loss = None

            if self.runai.step % self.runai.steps == self.runai.steps - 1:
                loss = self.__class__.__optimizer__.step(self, closure)
            else:
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure() # TODO(levosos): is this good?

            self.runai.step += 1
            return loss

        def zero_grad(self):
            if self.runai.step % self.runai.steps == 0:
                self.__class__.__optimizer__.zero_grad(self)

        @property
        def runai(self):
            return self.state['runai']

    # the main idea is to dynamically create a class that has all the functionality of the passed optimizer
    # (this is done by inheriting it) while overriding `step()` and `zero_grad()` to accumulate the gradients
    # and actually assign and zero them once in a few steps
    d = dict(_GradientAccumulationOptimizer.__dict__)
    d['__optimizer__'] = optimizer.__class__

    cls = type(
        optimizer.__class__.__name__,
        (optimizer.__class__,),
        d
    )
    return cls(steps, optimizer.param_groups)

# declare a GA version of builtin optimizers
def _optimizer(optimizer):
    setattr(
        sys.modules[__name__],
        optimizer,
        lambda steps, *args, **kwargs: Optimizer(getattr(torch.optim, optimizer)(*args, **kwargs), steps)
    )

[_optimizer(optimizer) for optimizer in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'SGD', 'Rprop', 'RMSprop']]
