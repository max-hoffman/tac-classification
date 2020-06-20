#!/bin/bash

set -xeou pipefail

DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
BASE=$DIR/..

cd $BASE
docker run --rm -p 8888:8888 -v $(pwd):/home/jovyan/work tac-jupyter:latest
