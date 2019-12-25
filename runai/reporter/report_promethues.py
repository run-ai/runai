from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway
from os import environ

GROUPING_KEY = "podUUID"
GATEWAY_URL_KEY = "reporterGatewayURL"
PUSH_GATEWAY_JOB_NAME = "reporter_pod_info"
REPORTER_PUSH_GATEWAY_METRIC_PREFIX = "reporter_push_gateway_metric"
REPORTER_PUSH_GATEWAY_METRIC_PARAMETER = "reporter_push_gateway_parameter"

def reportMetric(reporter_metric_name, reporter_metric_value):
    createGaugeAndPushToGateway(reporter_metric_name, reporter_metric_value)

def reportParameter(reporter_param_name, reporter_param_value):
    createGaugeAndPushToGateway(reporter_param_name, reporter_param_value)

def createGaugeAndPushToGateway(reporter_name, reporter_value, isMetric=True):
    label_names = []
    label_values = []
    gauge_value = None
    gauge_name = ""
    registry = CollectorRegistry()

    if isMetric:
        label_names = ['metric_name', 'push_gateway_type']
        label_values = [reporter_name, 'metric']
        gauge_name = REPORTER_PUSH_GATEWAY_METRIC_PREFIX + "_" + reporter_name
    else:
        label_names = ['param_name', 'param_value', 'push_gateway_type']
        label_values = [reporter_name, reporter_value, 'parameter']
        gauge_name = REPORTER_PUSH_GATEWAY_METRIC_PARAMETER + "_" + reporter_name

    gauge = Gauge(name=gauge_name, documentation="",labelnames=label_names, registry=registry)


    if isMetric:
        label_names = ['metric_name', 'push_gateway_type']
        label_values = [reporter_name, 'metric']
        gauge_name = REPORTER_PUSH_GATEWAY_METRIC_PREFIX + "_" + reporter_name
        gauge_value = reporter_value
    else:
        label_names = ['param_name', 'param_value', 'push_gateway_type']
        label_values = [reporter_name, reporter_value, 'parameter']
        gauge_name = REPORTER_PUSH_GATEWAY_METRIC_PARAMETER + "_" + reporter_name
        gauge_value = 1

    gauge.labels(*label_values).set(gauge_value)

    pushadd_to_gateway(gateway=environ[GATEWAY_URL_KEY], job=PUSH_GATEWAY_JOB_NAME,
                       registry=registry, grouping_key={GROUPING_KEY: environ[GROUPING_KEY]})
