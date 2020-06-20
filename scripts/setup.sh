#!/bin/bash

set -xeou pipefail

DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
BASE=$DIR/..

mkdir -p $BASE/data/{train_input, train_output}

cd $BASE/docker
docker build -t tac-jupyter -f Dockerfile.jupyter .
docker build -t tac-shell -f Dockerfile.shell .

