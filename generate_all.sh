# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


#!/bin/bash
mkdir -p sbatch
./generate_antmaze.sh $1 $2 $3
./generate_adroit.sh $1 $2 $3
./generate_cog.sh $1 $2 $3
