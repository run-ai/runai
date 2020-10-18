FROM nvidia/cuda:10.0-runtime-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        libcudnn7=7.6.0.64-1+cuda10.0 \
        python3.6 \
        python3.6-distutils \
        curl

RUN ln -s /usr/bin/python3.6 /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# install the sample script dependencies
RUN pip install \
        tensorflow-gpu==1.14.0 \
        Keras==2.2.4 \
        scipy==1.2.0 \
        pillow \
        coloredlogs

# install the Run:AI reporter Python library dependencies
RUN pip install prometheus_client==0.7.1

# add this entire repository to the docker image
ADD . /src

# configure the Python path to be aware of the Python
# library as if it was installed using `pip install`
ENV PYTHONPATH=/src

# configure the script entry point
WORKDIR /src
ENTRYPOINT ["python", "-u", "/src/examples/reporter/main.py"]
