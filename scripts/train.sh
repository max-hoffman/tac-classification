#!/bin/bash

set -xeou pipefail

DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
BASE=$DIR/..

cd $BASE
docker run \
    --rm \
    -v $(pwd):/home/ \
    tac-shell:latest \
    snakemake all -j -s /home/Snakefile -d /home
