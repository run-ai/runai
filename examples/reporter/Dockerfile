FROM python:3.7.0

# install the sample script dependencies
RUN pip install tensorflow==1.14 Keras==2.2.4

# install the Run:AI reporter Python library dependencies
RUN pip install prometheus_client==0.7.1

# add this entire repository to the docker image
ADD . /src

# configure the Python path to be aware of the Python
# library as if it was installed using `pip install`
ENV PYTHONPATH=/src

# configure the script entry point
ENTRYPOINT ["python", "-u", "/src/examples/reporter/keras_reporter_usage.py"]
