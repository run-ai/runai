FROM ubuntu:latest

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && pip install tensorflow  \
  && pip install keras

RUN pip3 install prometheus_client

RUN mkdir -p /workload

ADD . /workload

WORKDIR /workload
ENTRYPOINT ["examples/reporterlib/docker/run.sh"]
