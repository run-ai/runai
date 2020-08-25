import datetime
import enum
import itertools
import os
import random
import signal
import sys

import yaml

import runai.utils

class Status:
    Unassigned = 'unassigned'
    Assigned = 'assigned'
    Preempted = 'preempted'

class Strategy(enum.Enum):
    GridSearch = 0
    RandomSearch = 1

def init(root, subdir=None):
    """ Initialize Run:AI HPO Assistance

    arguments:
      - root: The root shared directory (most commonly over NFS). The root directory must exist.
      - subdir (optional): If passed, a subdirectory with this name will be created under `root`.
    """

    if subdir is not None:
        root = os.path.join(root, subdir)

        expected = FileExistsError if sys.version_info[0] == 3 else OSError
        try:
            os.mkdir(root)
            runai.utils.log.debug('Created experiment directory at %s', root)
        except expected: pass # another experiment has already created it

    flock = runai.utils.Flock(os.path.join(root, 'runai.yaml.lock'))
    path = os.path.join(root, 'runai.yaml')

    # place variables in this module for future use
    setattr(sys.modules[__name__], 'flock', flock)
    setattr(sys.modules[__name__], 'path', path)

    def handler(signum, frame):
        runai.utils.log.warning('Experiment was preempted (received SIGTERM)')

        experiment = getattr(sys.modules[__name__], 'experiment', None)
        if experiment:
            id = experiment['id']
            runai.utils.log.debug('Marking experiment #%d as preempted in YAML', id)

            with flock:
                with open(path, 'r') as f:
                    data = yaml.load(f, Loader=yaml.FullLoader)

                experiments = [e for e in data['experiments'] if e['id'] == id]
                assert len(experiments) == 1, 'Could not find experiment #%d in YAML' % id
                experiment = experiments[0]

                # mark the experiment as preempted
                experiment['status'] = Status.Preempted

                # set the experiment and the global last modification times to now
                now = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
                experiment['modificationTime'] = now
                data['modificationTime'] = now

                # rewrite the YAML
                with open(path, 'w') as f:
                    yaml.dump(data, f)

        # terminate the process (as it's currently in teardown)
        exit(1)

    signal.signal(signal.SIGTERM, handler)

def pick(grid, strategy):
    flock = getattr(sys.modules[__name__], 'flock')
    path = getattr(sys.modules[__name__], 'path')

    with flock:
        if not os.path.isfile(path): # the first experiment will create the YAML under the lock
            runai.utils.log.debug('Creating HPO YAML file at %s', path)

            data = dict(
                creationTime=datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            )

            if strategy == Strategy.GridSearch: # in grid search we generate all experiments in advance
                parameters = []
                iterables = []
                for parameter in grid:
                    parameters.append(parameter)
                    iterables.append(grid[parameter])

                experiments = []
                for combination in itertools.product(*iterables):
                    experiment = dict(
                        id=len(experiments) + 1,
                        status=Status.Unassigned,
                        config={ parameters[j]: combination[j] for j in range(len(parameters)) },
                    )
                    experiments.append(experiment)

                data['experiments'] = experiments

            with open(path, 'w') as f:
                yaml.dump(data, f)

        # the YAML file now exists for sure

        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        if strategy == Strategy.GridSearch: # pick an unassigned or a preempted experiment
            experiment = None
            for prospect in data['experiments']:
                if prospect['status'] in [Status.Unassigned, Status.Preempted]:
                    experiment = prospect
                    break
            assert experiment is not None, 'Could not find an unassigned HPO experiment'
            runai.utils.log.info('Picked %sHPO experiment #%d with configuration %s', ('preempted ' if experiment['status'] == Status.Preempted else ''), experiment['id'], experiment['config'])
        elif strategy == Strategy.RandomSearch: # continue a preempted experiment or randomize a new one
            if 'experiments' not in data:
                data['experiments'] = []

            # first check if there are any preempted experiments that should be continued
            preempted = [experiment for experiment in data['experiments'] if experiment['status'] == Status.Preempted]

            if len(preempted) > 0: # if there are, continue the first one
                experiment = preempted[0]
                runai.utils.log.info('Continuing preempted HPO experiment #%d with configuration %s', experiment['id'], experiment['config'])
            else:
                experiment = dict(
                    id=1 if len(data['experiments']) == 0 else (max([experiment['id'] for experiment in data['experiments']]) + 1),
                    config={ parameter: random.choice(grid[parameter]) for parameter in grid },
                )
                runai.utils.log.info('Randomized HPO experiment #%d with configuration %s', experiment['id'], experiment['config'])
                data['experiments'].append(experiment)
        else:
            raise ValueError('Unrecognized strategy %s' % str(strategy))

        # mark the experiment as assigned
        experiment['status'] = Status.Assigned

        # set the experiment and the global last modification times to now
        now = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        experiment['modificationTime'] = now
        data['modificationTime'] = now

        # rewrite the YAML
        with open(path, 'w') as f:
            yaml.dump(data, f)

    # save the experiment for later use
    setattr(sys.modules[__name__], 'experiment', experiment)

    return experiment['config']

def report(epoch, metrics):
    flock = getattr(sys.modules[__name__], 'flock')
    path = getattr(sys.modules[__name__], 'path')
    id = getattr(sys.modules[__name__], 'experiment')['id']

    with flock:
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        experiments = [e for e in data['experiments'] if e['id'] == id]
        assert len(experiments) == 1, 'Could not find experiment #%d in YAML' % id
        experiment = experiments[0]

        if 'reports' not in experiment:
            experiment['reports'] = []

        experiment['reports'].append(dict(
            epoch=epoch,
            metrics=metrics,
            reportTime=datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        ))

        # rewrite the YAML
        with open(path, 'w') as f:
            yaml.dump(data, f)
