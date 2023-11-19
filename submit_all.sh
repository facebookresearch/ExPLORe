# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

mkdir logs/out/ -p
mkdir logs/err/ -p

sbatch sbatch/antmaze-main.sh
sbatch sbatch/antmaze-bc-jsrl.sh
sbatch sbatch/antmaze-bc.sh
sbatch sbatch/antmaze-online.sh
sbatch sbatch/antmaze-oracle.sh
sbatch sbatch/antmaze-min.sh

sbatch sbatch/adroit-main.sh
sbatch sbatch/adroit-main-relocate-reset.sh
sbatch sbatch/adroit-bc-jsrl.sh
sbatch sbatch/adroit-bc.sh
sbatch sbatch/adroit-min.sh
sbatch sbatch/adroit-online.sh
sbatch sbatch/adroit-oracle.sh

sbatch sbatch/cog-bc-jsrl.sh
sbatch sbatch/cog-bc.sh
sbatch sbatch/cog-main-ours.sh
sbatch sbatch/cog-main-oracle.sh
sbatch sbatch/cog-min.sh
sbatch sbatch/cog-online.sh
