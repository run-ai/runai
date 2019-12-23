from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway
from os import environ

GROUPING_KEY = "podUUID"
GATEWAY_URL_KEY = "reporterGatewayURL"
PUSH_GATEWAY_JOB_NAME = "reporter_pod_info"
REPORTER_PUSH_GATEWAY_METRIC_PREFIX = "reporter_push_gateway_metric"
REPORTER_PUSH_GATEWAY_METRIC_PARAMETER = "reporter_push_gateway_parameter"

def reportMetric(reporter_metric_name, reporter_metric_value):
    registry = CollectorRegistry()
    metric_label_names = ['metric_name', 'push_gateway_type']
    metric_label_values = [reporter_metric_name, 'metric']

    gauge_name = REPORTER_PUSH_GATEWAY_METRIC_PREFIX + "_" + reporter_metric_name
    gauge = Gauge(name=gauge_name, documentation="",
                  labelnames=metric_label_names, registry=registry)
    gauge.labels(*metric_label_values).set(reporter_metric_value)

    pushadd_to_gateway(gateway=environ[GATEWAY_URL_KEY], job=PUSH_GATEWAY_JOB_NAME,
                       registry=registry, grouping_key={GROUPING_KEY: environ[GROUPING_KEY]})

def reportParameter(reporter_param_name, reporter_param_value):
    registry = CollectorRegistry()
    metric_label_names = ['param_name', 'param_value', 'push_gateway_type']
    metric_label_values = [reporter_param_name, reporter_param_value, 'parameter']

    gauge_name = REPORTER_PUSH_GATEWAY_METRIC_PARAMETER + "_" + reporter_param_name
    gauge = Gauge(name=gauge_name, documentation="",
                  labelnames=metric_label_names, registry=registry)
    gauge.labels(*metric_label_values).set(1)
    pushadd_to_gateway(gateway=environ[GATEWAY_URL_KEY], job=PUSH_GATEWAY_JOB_NAME,
                       registry=registry, grouping_key={GROUPING_KEY: environ[GROUPING_KEY]})
