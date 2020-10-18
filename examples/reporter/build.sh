#!/bin/sh

# this script should be called from the very root of this repository

if [ ! -d "./examples/reporter/cifar-10" ]; then
    echo "Downloading CIFAR10 dataset"
    wget -nc https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar zxf cifar-10-python.tar.gz
    mv cifar-10-batches-py ./examples/reporter/cifar-10
    rm cifar-10-python.tar.gz
fi

docker build -f examples/reporter/Dockerfile -t runai/example-python-library-reporter .
