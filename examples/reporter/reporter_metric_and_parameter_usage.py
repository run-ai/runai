from runai.reporter import reportMetric, reportParameter
from time import sleep

for step in range(1000):
    reportMetric("step", step)
    reportMetric("accuracy", step)
    reportMetric("loss", step)
    reportMetric("epoch", step)
    reportMetric("this_is_a_test", 123)
    reportParameter("state", "running")
    sleep(1)
