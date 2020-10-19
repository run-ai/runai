import enum
import multiprocessing
import os

from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway

import runai.utils

GROUPING_KEY = "podUUID"
GATEWAY_URL_KEY = "reporterGatewayURL"
PUSH_GATEWAY_JOB_NAME = "reporter_pod_info"
REPORTER_PUSH_GATEWAY_METRIC_PREFIX = "reporter_push_gateway_metric"
REPORTER_PUSH_GATEWAY_METRIC_PARAMETER = "reporter_push_gateway_parameter"
REPORTER = None
RETRIES = 3

class ReportType(enum.Enum):
    metric = 1
    parameter = 2

def reportMetric(name, value):
    send(name, value, ReportType.metric)

def reportParameter(name, value):
    send(name, value, ReportType.parameter)

def push(reporter_name, reporter_value, report_type):
    if push.failures >= RETRIES:
        return

    registry = CollectorRegistry()

    if report_type is ReportType.metric:
        label_names = ['metric_name', 'push_gateway_type']
        label_values = [reporter_name, 'metric']
        gauge_name = REPORTER_PUSH_GATEWAY_METRIC_PREFIX + "_" + reporter_name
        gauge_value = reporter_value

    else:
        label_names = ['param_name', 'param_value', 'push_gateway_type']
        label_values = [reporter_name, reporter_value, 'parameter']
        gauge_name = REPORTER_PUSH_GATEWAY_METRIC_PARAMETER + "_" + reporter_name
        gauge_value = 1

    gauge = Gauge(name=gauge_name, documentation="",labelnames=label_names, registry=registry)

    gauge.labels(*label_values).set(gauge_value)

    try:
        pushadd_to_gateway(gateway=os.environ[GATEWAY_URL_KEY], job=PUSH_GATEWAY_JOB_NAME,
                        registry=registry, grouping_key={GROUPING_KEY: os.environ[GROUPING_KEY]})

        push.failures = 0
    except IOError as e:
        runai.utils.log.error('Failed pushing registry to push gateway (%s)', e)
        push.failures += 1

push.failures = 0

class Reporter(multiprocessing.Process):
    @staticmethod
    def _impl(queue):
        while True:
            msg = queue.get()

            if msg is None:
                # a `None` message tells us that the job is done
                # successfully and we can finish running
                break

            # we catch all errors as this process should
            # be running until told to stop
            try:
                push(*msg)
            except:
                pass

    def __init__(self):
        # create a shared queue for this process and the worker process
        self._queue = multiprocessing.Queue()

        # initialize the process and pass the shared queue
        super(Reporter, self).__init__(target=Reporter._impl, args=(self._queue,))

    def start(self, daemon):
        self.daemon = daemon
        super(Reporter, self).start()

        runai.utils.log.debug('Started reporting worker process (%d)', self.pid)

    def finish(self):
        # tell the worker process to finish
        self.send(None)

        # wait for it to finish
        self.join()

        runai.utils.log.debug('Reporting process (%d) finished', self.pid)

    def __enter__(self):
        # when running in a managed context (i.e. using Python `with` keyword),
        # we will gracefully finish the process so we don't set the process
        # as a daemon
        self.start(daemon=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def send(self, msg):
        self._queue.put(msg)

    def reportMetric(self, name, value):
        send((name, value, ReportType.metric))

    def reportParameter(self, name, value):
        send((name, value, ReportType.parameter))

def send(*args):
    global REPORTER

    if REPORTER is None:
        REPORTER = Reporter()

        # in general, processes have to be joined at the end of this process.
        # we don't want to have this as a fundamental requirement for using this
        # library, so we set the process daemon flag.
        # this will cause the worker process to be automatically terminated when
        # this process will end, and this allows users to not explicitly call `finish`.
        # it's still recommended to call `finish` to ensure all reports have been
        # successfully processed and pushed
        REPORTER.start(daemon=True)

    REPORTER.send(args)

def finish():
    global REPORTER

    if REPORTER is not None:
        REPORTER.finish()
        REPORTER = None
