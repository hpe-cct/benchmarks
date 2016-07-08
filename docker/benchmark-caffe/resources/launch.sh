#!/bin/bash

set -e

net=${1:-alexnet}
batch_size=${2:-128}

# Supported networks:
# "alexnet" - /opt/caffe/models/bvlc_alexnet/deploy.prototxt (dim: 10)
# "cifar10_quick" - /opt/caffe/examples/cifar10/cifar10_quick.prototxt (dim: 1)

case $net in
  "alexnet")
  path="/opt/caffe/models/bvlc_alexnet/deploy.prototxt"
  ;;
  "cifar10_quick")
  path="/opt/caffe/examples/cifar10/cifar10_quick.prototxt"
  ;;
  *)
  echo "No mapping for network \"$net\", aborting." >&2
  exit 1
esac

echo "Net: $net" >&2
echo "Net path: $path" >&2
echo "Batch size: $batch_size" >&2

# replace the first instance of "dim: N" with "dim: $batch_size"
sed -i "s/dim: [[:digit:]]\+/dim: $batch_size/1" ${path}

# turn execution over to Caffe
exec /opt/caffe/build/tools/caffe time -model ${path} -gpu all
