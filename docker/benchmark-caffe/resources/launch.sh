#!/bin/bash

# (c) Copyright 2016 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
exec /opt/caffe/build/tools/caffe time -model ${path} -gpu all 1>&2
