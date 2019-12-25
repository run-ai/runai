# Build the Dockerfile from the root directory repository
FROM ubuntu:18.04

RUN apt-get update \
    && apt-get install -y python3-pip python3-dev

RUN cd /usr/local/bin \
  && ln -s /usr/bin/python3 python

RUN pip3 install --upgrade pip \
  && pip install tensorflow==1.14 \
  && pip install keras==2.2.4

RUN pip3 install prometheus_client==0.7.1

RUN mkdir -p /workload

ADD . /workload

WORKDIR /workload
ENTRYPOINT ["examples/reporter/docker/run.sh"]
